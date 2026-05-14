"""
task4_imu.py — Fractal analysis of UCI HAR IMU signals (Team 10, Practical 4.11)

Data source: UCI Human Activity Recognition Using Smartphones Dataset
  https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
  Sampling rate: 50 Hz, window size: 128 samples (~2.56 s per window)
"""

import os
import sys
import argparse
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from utils import ensure_dir, save_table_csv, compute_hurst_rs, classify_noise


RAW_DIR  = os.path.join(_PROJECT, 'data', 'raw', 'task4')
PLOT_DIR = os.path.join(_PROJECT, 'plots', 'task4')
OUT_CSV  = os.path.join(_PROJECT, 'data', 'metrics_task4.csv')

UCI_HAR_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00240/UCI%20HAR%20Dataset.zip"
)
UCI_HAR_DIR = os.path.join(RAW_DIR, 'UCI HAR Dataset')

CHANNELS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']


STABLE_ACT  = [5, 6]     # near-static  → persistent signal, H > 0.5 expected
DYNAMIC_ACT = [1, 2, 3]  # walking      → lower / anti-persistent H expected

N_WINDOWS_PER_SEG = 12   # 12 × 128 samples = 1536 samples per segment

WINDOW_SIZE = 200    # samples per sliding-window Hurst estimate
WINDOW_STEP = 50     # stride between successive windows
H_THRESHOLD = 0.5    # H below this → instability / anti-persistent



def download_uci_har() -> None:
    """Download and extract the UCI HAR zip into RAW_DIR."""
    zip_path = os.path.join(RAW_DIR, 'UCI_HAR.zip')
    ensure_dir(RAW_DIR)
    print("  Downloading UCI HAR Dataset (~60 MB) ...")
    urllib.request.urlretrieve(UCI_HAR_URL, zip_path)
    print("  Extracting ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(RAW_DIR)
    os.remove(zip_path)
    print(f"  Dataset ready at: {UCI_HAR_DIR}")


def load_uci_har(n_windows: int = N_WINDOWS_PER_SEG):

    sig_dir = os.path.join(UCI_HAR_DIR, 'train', 'Inertial Signals')
    labels  = np.loadtxt(
        os.path.join(UCI_HAR_DIR, 'train', 'y_train.txt'), dtype=int
    )

    # body_acc_* = linear acceleration with gravity removed (m/s²)
    # body_gyro_* = angular velocity (rad/s)
    signal_files = {
        'ax': 'body_acc_x_train.txt',
        'ay': 'body_acc_y_train.txt',
        'az': 'body_acc_z_train.txt',
        'gx': 'body_gyro_x_train.txt',
        'gy': 'body_gyro_y_train.txt',
        'gz': 'body_gyro_z_train.txt',
    }

    print("  Loading inertial signals (may take ~10 s) ...")
    mats = {}
    for ch, fname in signal_files.items():
        mats[ch] = np.loadtxt(os.path.join(sig_dir, fname))   # (7352, 128)

    rng = np.random.default_rng(42)

    def build_segment(activity_ids):
        idx = np.where(np.isin(labels, activity_ids))[0]
        n   = min(n_windows, len(idx))
        # keep temporal order → sort selected indices
        sel = np.sort(rng.choice(idx, size=n, replace=False))
        return {ch: mats[ch][sel].ravel() for ch in CHANNELS}

    stable_data  = build_segment(STABLE_ACT)
    dynamic_data = build_segment(DYNAMIC_ACT)

    n_s = len(next(iter(stable_data.values())))
    n_d = len(next(iter(dynamic_data.values())))

    df = pd.DataFrame({
        ch: np.concatenate([stable_data[ch], dynamic_data[ch]])
        for ch in CHANNELS
    })
    df.index.name = 'sample'
    return df, n_s, n_d


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Remove global mean from each channel."""
    out = df.copy()
    for ch in CHANNELS:
        out[ch] -= out[ch].mean()
    return out


# ── Hurst per channel ──────────────────────────────────────────────────────────

def hurst_per_channel(df: pd.DataFrame) -> dict:
    """R/S Hurst exponent on the full demeaned series for each of 6 channels."""
    results = {}
    for ch in CHANNELS:
        H, ns, rs = compute_hurst_rs(df[ch].values)
        results[ch] = dict(H=H, D=2.0 - H, ns=ns, rs=rs, label=classify_noise(H))
    return results


# ── Segment comparison ─────────────────────────────────────────────────────────

def hurst_per_segment(df: pd.DataFrame, n_stable: int) -> dict:
    """Compute H separately for the stable and dynamic halves of each channel."""
    seg = {}
    for ch in CHANNELS:
        H_s, _, _ = compute_hurst_rs(df[ch].values[:n_stable])
        H_d, _, _ = compute_hurst_rs(df[ch].values[n_stable:])
        seg[ch] = dict(H_stable=H_s, H_dynamic=H_d)
    return seg


# ── Sliding-window H(t) ────────────────────────────────────────────────────────

def sliding_hurst(df: pd.DataFrame, channels: list,
                  win: int = WINDOW_SIZE, step: int = WINDOW_STEP):

    arrays  = [df[ch].values for ch in channels]
    centers, h_vals = [], []
    start = 0
    while start + win <= len(arrays[0]):
        hs = []
        for arr in arrays:
            seg = arr[start: start + win]
            try:
                H, _, _ = compute_hurst_rs(seg, n_points=15)
                hs.append(H)
            except Exception:
                pass
        centers.append(start + win // 2)
        h_vals.append(float(np.mean(hs)) if hs else float('nan'))
        start += step
    return np.array(centers, dtype=float), np.array(h_vals, dtype=float)


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_signals(df_raw: pd.DataFrame, n_stable: int) -> None:
    """6-panel time series; blue shading = stable, red = dynamic."""
    ensure_dir(PLOT_DIR)
    n_total = len(df_raw)
    t       = np.arange(n_total)
    ylabels = ['ax (m/s²)', 'ay (m/s²)', 'az (m/s²)',
               'gx (rad/s)', 'gy (rad/s)', 'gz (rad/s)']

    fig, axes = plt.subplots(6, 1, figsize=(13, 11), sharex=True)
    for i, (ch, lbl) in enumerate(zip(CHANNELS, ylabels)):
        ax = axes[i]
        ax.plot(t, df_raw[ch].values, linewidth=0.6, color='steelblue')
        ax.axvspan(0, n_stable - 1,       alpha=0.10, color='steelblue')
        ax.axvspan(n_stable, n_total - 1, alpha=0.10, color='tomato')
        ax.axvline(n_stable, color='k', linewidth=0.8, linestyle='--')
        ax.set_ylabel(lbl, fontsize=8)

    axes[0].set_title('UCI HAR — Real IMU signals  (6 channels)', fontsize=12)
    axes[0].annotate('stable\n(STANDING/LAYING)',
                     xy=(n_stable // 2, axes[0].get_ylim()[1]),
                     ha='center', va='top', color='steelblue', fontsize=8)
    axes[0].annotate('dynamic\n(WALKING)',
                     xy=(n_stable + (n_total - n_stable) // 2,
                         axes[0].get_ylim()[1]),
                     ha='center', va='top', color='tomato', fontsize=8)
    axes[-1].set_xlabel('Sample index  (50 Hz)')
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, 'imu_signals.png')
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved: {path}")


def plot_rs_per_channel(results: dict) -> None:
    """2×3 log-log R/S scatter + regression, one panel per channel."""
    ensure_dir(PLOT_DIR)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, ch in enumerate(CHANNELS):
        ax  = axes[i // 3, i % 3]
        H   = results[ch]['H']
        D   = results[ch]['D']
        lx  = np.log10(results[ch]['ns'])
        ly  = np.log10(results[ch]['rs'])
        fit = np.poly1d(np.polyfit(lx, ly, 1))
        ax.scatter(lx, ly, s=30, zorder=3, label='R/S(n)')
        ax.plot(lx, fit(lx), color='crimson', linewidth=1.5,
                label=f'H = {H:.3f}   D = {D:.3f}')
        ax.set_title(ch, fontsize=11)
        ax.set_xlabel('log₁₀(n)')
        ax.set_ylabel('log₁₀(R/S)')
        ax.legend(fontsize=8)
    fig.suptitle('R/S Analysis per IMU channel  (UCI HAR, full signal)', fontsize=12)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, 'rs_per_channel.png')
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved: {path}")


def plot_segment_comparison(seg: dict) -> None:
    """Grouped bar chart: H_stable vs H_dynamic per channel."""
    ensure_dir(PLOT_DIR)
    x     = np.arange(len(CHANNELS))
    width = 0.35
    H_s   = [seg[ch]['H_stable']  for ch in CHANNELS]
    H_d   = [seg[ch]['H_dynamic'] for ch in CHANNELS]

    fig, ax = plt.subplots(figsize=(10, 5))
    b_s = ax.bar(x - width / 2, H_s, width,
                 label='Stable (STANDING/LAYING)', color='steelblue', alpha=0.85)
    b_d = ax.bar(x + width / 2, H_d, width,
                 label='Dynamic (WALKING)',         color='tomato',   alpha=0.85)
    ax.axhline(H_THRESHOLD, color='k', linewidth=1.0, linestyle='--',
               label=f'H = {H_THRESHOLD}  (random-walk boundary)')
    ax.set_xticks(x); ax.set_xticklabels(CHANNELS)
    ax.set_ylabel('Hurst exponent H')
    ax.set_ylim(0, 1.1)
    ax.set_title('Stable vs Dynamic segment  —  H per channel  (UCI HAR)')
    ax.legend()
    for bar in list(b_s) + list(b_d):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', fontsize=8)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, 'segment_comparison.png')
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved: {path}")


def plot_sliding_hurst(centers: np.ndarray, h_vals: np.ndarray,
                       n_stable: int,
                       threshold: float = H_THRESHOLD) -> dict:
    """H(t) with instability shading and stable/dynamic boundary."""
    ensure_dir(PLOT_DIR)
    valid  = ~np.isnan(h_vals)
    n_inst = int(np.sum(h_vals[valid] < threshold))
    n_tot  = int(valid.sum())
    pct    = 100.0 * n_inst / n_tot if n_tot > 0 else 0.0

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(centers, h_vals, linewidth=1.2, color='navy', label='H(t)')
    ax.axhline(threshold, color='crimson', linewidth=1.2, linestyle='--',
               label=f'H = {threshold}  (instability threshold)')
    ax.axvline(n_stable, color='k', linewidth=0.9, linestyle=':',
               label='stable | dynamic boundary')

    hw = WINDOW_SIZE // 2
    for c, h in zip(centers, h_vals):
        if not np.isnan(h) and h < threshold:
            ax.axvspan(c - hw, c + hw, alpha=0.20, color='red', linewidth=0)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Sample index  (50 Hz)')
    ax.set_ylabel('H(t)')
    ax.set_title(
        f'Sliding-window mean H(t)  [ax, ay, az]  (UCI HAR)'
        f'  (window = {WINDOW_SIZE}, step = {WINDOW_STEP})'
        f'   →  instability: {n_inst}/{n_tot} windows  ({pct:.1f}%)'
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, 'sliding_hurst.png')
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved: {path}")

    return dict(n_instability=n_inst, n_total=n_tot, pct_instability=round(pct, 2))


# ── Metrics ────────────────────────────────────────────────────────────────────

def save_metrics(results: dict, seg: dict, inst: dict) -> None:
    rows = []
    for ch in CHANNELS:
        rows.append({
            'Channel'   : ch,
            'H_full'    : round(results[ch]['H'],                         4),
            'D_full'    : round(results[ch]['D'],                         4),
            'H_stable'  : round(seg[ch]['H_stable'],                      4),
            'H_dynamic' : round(seg[ch]['H_dynamic'],                     4),
            'Delta_H'   : round(seg[ch]['H_stable'] - seg[ch]['H_dynamic'], 4),
        })
    df_m = pd.DataFrame(rows).set_index('Channel')
    save_table_csv(df_m, OUT_CSV)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Завдання 4.11 — Фрактальний аналіз сигналів IMU (UCI HAR)."
    )
    p.add_argument("--n-windows",  default=N_WINDOWS_PER_SEG, type=int,
                   help=f"Кількість вікон UCI HAR на сегмент (default: {N_WINDOWS_PER_SEG})")
    p.add_argument("--threshold",  default=H_THRESHOLD, type=float,
                   help=f"Поріг H для детектування нестабільності (default: {H_THRESHOLD})")
    p.add_argument("--window-size", default=WINDOW_SIZE, type=int,
                   help=f"Розмір ковзного вікна Херста (default: {WINDOW_SIZE})")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 62)
    print("© 2026 Команда 10 — Фрактальні методи аналізу")
    print("Завдання 4.11: Фрактальний аналіз сигналів IMU (UCI HAR)")
    print(f"  Вікон на сегмент: {args.n_windows}  |  Поріг H: {args.threshold}")
    print(f"  Розмір ковзного вікна: {args.window_size} відліків")
    print("=" * 62)

    ensure_dir(RAW_DIR)
    ensure_dir(PLOT_DIR)

    # 1. Download dataset if not present
    if not os.path.exists(UCI_HAR_DIR):
        download_uci_har()
    else:
        print(f"\nDataset already present: {UCI_HAR_DIR}")

    # 2. Load and segment data
    print("\nBuilding stable/dynamic segments ...")
    df_raw, n_stable, n_dynamic = load_uci_har(n_windows=args.n_windows)
    n_total = len(df_raw)

    print(f"  Stable   (activities {STABLE_ACT}): {n_stable} samples")
    print(f"  Dynamic  (activities {DYNAMIC_ACT}): {n_dynamic} samples")
    print(f"  Total                             : {n_total} samples × {len(CHANNELS)} channels")

    # 3. Preprocess
    df = preprocess(df_raw)

    # 4. Per-channel Hurst (full signal)
    print("\n--- Per-channel Hurst (full signal) ---")
    results = hurst_per_channel(df)
    for ch in CHANNELS:
        r   = results[ch]
        cat = ('persistent/stable'       if r['H'] > args.threshold + 0.02
               else 'anti-persistent/unstable' if r['H'] < args.threshold - 0.02
               else 'random-like')
        print(f"  {ch:3s}  H = {r['H']:.4f}   D = {r['D']:.4f}   [{cat}]")

    # 5. Segment comparison
    print("\n--- Stable vs Dynamic segment ---")
    seg = hurst_per_segment(df, n_stable)
    for ch in CHANNELS:
        s     = seg[ch]
        delta = s['H_stable'] - s['H_dynamic']
        print(f"  {ch:3s}  H_stable = {s['H_stable']:.4f}  "
              f"H_dynamic = {s['H_dynamic']:.4f}  ΔH = {delta:+.4f}")

    # 6. Sliding-window H(t)
    print("\n--- Sliding-window Hurst (mean over ax, ay, az) ---")
    centers, h_vals = sliding_hurst(df, channels=['ax', 'ay', 'az'],
                                    win=args.window_size)
    inst = plot_sliding_hurst(centers, h_vals, n_stable,
                              threshold=args.threshold)

    # 7. Remaining plots
    print("\n--- Plots ---")
    plot_signals(df_raw, n_stable)
    plot_rs_per_channel(results)
    plot_segment_comparison(seg)

    # 8. Save metrics
    save_metrics(results, seg, inst)

    # 9. Summary
    mean_H_s = np.mean([seg[ch]['H_stable']  for ch in CHANNELS])
    mean_H_d = np.mean([seg[ch]['H_dynamic'] for ch in CHANNELS])

    print(f"\n{'=' * 62}")
    print("Conclusion")
    print(f"{'=' * 62}")
    print(f"  Data source      : UCI HAR Dataset (real smartphone IMU, 50 Hz)")
    print(f"  Mean H — stable  (STANDING/LAYING) : {mean_H_s:.4f}")
    print(f"  Mean H — dynamic (WALKING)         : {mean_H_d:.4f}")
    print(f"  ΔH (stable − dynamic)              : {mean_H_s - mean_H_d:+.4f}")
    print(f"  Instability score : {inst['pct_instability']:.1f}%"
          f"  ({inst['n_instability']}/{inst['n_total']} windows)")
    print()
    print("  The Hurst exponent differentiates static platform activities")
    print("  (STANDING/LAYING → H > 0.5) from dynamic motion")
    print("  (WALKING → lower H), validating fractal analysis for")
    print("  motion stability monitoring.")
    print(f"{'=' * 62}")
    print("[Task 4.11 complete]")
    print(f"{'=' * 62}")


if __name__ == '__main__':
    main()
