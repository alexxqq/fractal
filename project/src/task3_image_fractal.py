"""
task3_image_fractal.py — Fractal dimension / complexity analysis of images
(Team 10, Practical 3.10)

Parts:
  A. Box-counting fractal dimension  D_box
  B. Wavelet scale-energy complexity estimate  D_wavelet  (approximate)
  C. Noise robustness: both metrics vs Gaussian noise sigma

Images analysed:
  1. Sierpinski triangle  — known fractal reference (theoretical D ≈ 1.585)
  2. Natural texture      — skimage.data.grass / smoothed noise fallback
  3. Simple geometric     — circle on white background (low complexity)

All outputs saved to plots/task3/ and data/metrics_task3.csv.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from utils import ensure_dir, save_table_csv

PLOT_DIR  = os.path.join(_PROJECT, 'plots', 'task3')
DATA_DIR  = os.path.join(_PROJECT, 'data', 'raw', 'task3')
OUT_CSV   = os.path.join(_PROJECT, 'data', 'metrics_task3.csv')

# Gaussian noise sigmas for Part C (applied to [0,1]-scaled images)
NOISE_SIGMAS = [0.00, 0.01, 0.05, 0.10, 0.20]


# ── Image generation ──────────────────────────────────────────────────────────

def make_sierpinski(size: int = 512) -> np.ndarray:

    # round up to power of 2 so every box scale divides N exactly
    N = 1
    while N < size:
        N <<= 1
    N = min(N, 512)

    img   = np.ones((N, N), dtype=np.uint8) * 255   # white background

    r_idx = np.arange(N, dtype=np.int32)[:, None]   # (N, 1)  broadcast rows
    c_idx = np.arange(N, dtype=np.int32)[None, :]   # (1, N)  broadcast cols

    # c is a submask of r ⟺ Pascal triangle entry C(r, c) is odd
    mask = (c_idx <= r_idx) & ((r_idx & c_idx) == c_idx)
    img[mask] = 0   # black = foreground

    return img


def make_natural_texture(size: int = 512) -> np.ndarray:
    """
    Natural-looking texture.  Tries skimage.data.grass; falls back to
    band-limited Gaussian noise layered at multiple scales.
    """
    try:
        import skimage.data as skd
        from skimage.color import rgb2gray
        from skimage.transform import resize as sk_resize
        try:
            raw = skd.grass()
        except AttributeError:
            raw = skd.camera()
        arr = rgb2gray(raw) if raw.ndim == 3 else raw.astype(float)
        arr = sk_resize(arr, (size, size), anti_aliasing=True)
        return (arr * 255).astype(np.uint8)
    except Exception:
        pass

    # Fallback: layered multi-scale noise
    rng = np.random.default_rng(42)
    out = np.zeros((size, size), dtype=float)
    for sigma, amp in [(2, 1.0), (8, 0.5), (32, 0.25), (64, 0.1)]:
        out += amp * gaussian_filter(rng.standard_normal((size, size)), sigma)
    out -= out.min()
    out /= out.max()
    return (out * 255).astype(np.uint8)


def make_geometric(size: int = 512) -> np.ndarray:

    img       = np.ones((size, size), dtype=np.uint8) * 255
    cx = cy   = size // 2
    r         = size // 3
    thickness = max(3, size // 128)   # ~4 px wide at size=512
    ys, xs    = np.ogrid[:size, :size]
    dist      = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    img[(dist >= r - thickness) & (dist <= r + thickness)] = 0
    return img


def save_image_png(arr: np.ndarray, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    from PIL import Image
    Image.fromarray(arr).save(path)
    print(f"  Saved image: {path}")


# ── Part A — Box-counting ─────────────────────────────────────────────────────

def _to_binary(image: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale, then threshold at the image mean.
    Returns boolean array (True = foreground).
    """
    try:
        from skimage.color import rgb2gray
        gray = (rgb2gray(image) * 255).astype(np.uint8) if image.ndim == 3 else image.astype(np.uint8)
    except Exception:
        gray = image.astype(np.uint8)
    thresh = int(gray.mean())
    return gray < thresh    # dark pixels = foreground


def box_counting_dimension(image: np.ndarray,
                            box_sizes: list = None) -> tuple:

    binary = _to_binary(image)
    H, W   = binary.shape

    if box_sizes is None:
        max_exp  = int(np.floor(np.log2(min(H, W)))) - 1
        box_sizes = [2 ** k for k in range(1, max_exp + 1)]

    sizes, counts = [], []
    for l in box_sizes:
        if l >= min(H, W):
            continue
        rows = H // l
        cols = W // l
        # Reshape into (rows, l, cols, l) to vectorise box occupancy test
        patch    = binary[:rows * l, :cols * l].reshape(rows, l, cols, l)
        occupied = patch.any(axis=(1, 3))
        N        = int(occupied.sum())
        if N > 0:
            sizes.append(l)
            counts.append(N)

    sizes_arr  = np.array(sizes,  dtype=float)
    counts_arr = np.array(counts, dtype=float)

    log_inv_l = np.log(1.0 / sizes_arr)
    log_N     = np.log(counts_arr)
    D_box     = float(np.polyfit(log_inv_l, log_N, 1)[0])

    return D_box, sizes_arr, counts_arr


def plot_boxcount_loglog(sizes_arr, counts_arr, D_box, name: str) -> None:
    """Save log-log scatter N(l) vs 1/l with regression line."""
    ensure_dir(PLOT_DIR)
    log_inv_l = np.log(1.0 / sizes_arr)
    log_N     = np.log(counts_arr)
    fit       = np.poly1d(np.polyfit(log_inv_l, log_N, 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(log_inv_l, log_N, s=50, zorder=3, label='N(l)')
    ax.plot(log_inv_l, fit(log_inv_l), color='crimson', linewidth=1.5,
            label=f'Fit  D_box = {D_box:.4f}')
    ax.set_title(f'Box-counting  —  {name}')
    ax.set_xlabel('log(1/l)')
    ax.set_ylabel('log(N(l))')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, f'boxcount_loglog_{name}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Part B — Wavelet complexity ───────────────────────────────────────────────

def wavelet_complexity_dimension(image: np.ndarray,
                                  wavelet: str = 'db2',
                                  max_level: int = None) -> tuple:

    try:
        from skimage.color import rgb2gray
        gray = rgb2gray(image).astype(float) if image.ndim == 3 else image.astype(float)
    except Exception:
        gray = image.astype(float)
    if gray.max() > 1.5:
        gray /= 255.0

    try:
        import pywt
        if max_level is None:
            max_level = min(pywt.dwt_max_level(min(gray.shape), wavelet), 6)

        coeffs   = pywt.wavedec2(gray, wavelet, level=max_level)
        # coeffs[0] = approximation; coeffs[1:] = per-level detail tuples
        energies = []
        for detail in coeffs[1:]:
            LH, HL, HH = detail
            energies.append(float(np.sum(LH**2) + np.sum(HL**2) + np.sum(HH**2)))
        levels   = np.arange(1, len(energies) + 1, dtype=float)
        energies = np.array(energies, dtype=float)

    except ImportError:
        # Pyramid fallback: energy of successive Gaussian-blur differences
        max_level = max_level or 6
        prev = gray.copy()
        levels_list, energies_list = [], []
        for j in range(1, max_level + 1):
            blurred = gaussian_filter(prev, sigma=2)
            diff    = prev - blurred
            E       = float(np.sum(diff ** 2))
            if E > 0:
                levels_list.append(float(j))
                energies_list.append(E)
            prev = blurred
        levels   = np.array(levels_list)
        energies = np.array(energies_list)

    valid    = energies > 0
    lev_v    = levels[valid]
    eng_v    = energies[valid]

    if len(lev_v) < 2:
        return float('nan'), levels, energies

    log_scale = np.log(2.0 ** lev_v)
    log_E     = np.log(eng_v)
    beta      = float(np.polyfit(log_scale, log_E, 1)[0])
    D_wavelet = (7.0 - beta) / 2.0

    return D_wavelet, lev_v, eng_v


def plot_wavelet_energy(levels, energies, D_wavelet, name: str) -> None:
    """Save log(E_j) vs log(2^j) scatter + regression."""
    ensure_dir(PLOT_DIR)
    log_scale = np.log(2.0 ** levels)
    log_E     = np.log(energies)
    fit       = np.poly1d(np.polyfit(log_scale, log_E, 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(log_scale, log_E, s=50, zorder=3, label='E_j per level')
    ax.plot(log_scale, fit(log_scale), color='teal', linewidth=1.5,
            label=f'Fit  D_wavelet ≈ {D_wavelet:.4f}')
    ax.set_title(f'Wavelet scale-energy  —  {name}  (approx.)')
    ax.set_xlabel('log(2ʲ)  [scale]')
    ax.set_ylabel('log(E_j)  [detail energy]')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, f'wavelet_energy_{name}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Part C — Noise robustness ─────────────────────────────────────────────────

def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise (sigma on [0,1] scale) and clamp to uint8."""
    if sigma == 0.0:
        return image.copy()
    rng   = np.random.default_rng(0)
    noisy = image.astype(float) / 255.0
    noisy += rng.normal(0.0, sigma, noisy.shape)
    return (np.clip(noisy, 0.0, 1.0) * 255).astype(np.uint8)


def robustness_analysis(images: dict, sigmas=None) -> pd.DataFrame:
    """
    For each image × noise sigma: compute D_box and D_wavelet_approx.
    Returns a tidy DataFrame indexed by (Image, Sigma).
    """
    if sigmas is None:
        sigmas = NOISE_SIGMAS
    rows = []
    for name, img in images.items():
        print(f"  Robustness — {name} ...")
        for sigma in sigmas:
            noisy  = add_gaussian_noise(img, sigma)
            D_box, _, _ = box_counting_dimension(noisy)
            D_wav, _, _ = wavelet_complexity_dimension(noisy)
            rows.append(dict(
                Image=name, Sigma=sigma,
                D_box=round(D_box, 4),
                D_wavelet_approx=round(D_wav, 4),
            ))
    return pd.DataFrame(rows)


def plot_noise_robustness(df: pd.DataFrame) -> None:
    """Two-panel: D_box and D_wavelet vs noise sigma for all images."""
    ensure_dir(PLOT_DIR)
    names = df['Image'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name in names:
        sub = df[df['Image'] == name]
        axes[0].plot(sub['Sigma'], sub['D_box'],            'o-', label=name, linewidth=1.5)
        axes[1].plot(sub['Sigma'], sub['D_wavelet_approx'], 's-', label=name, linewidth=1.5)

    for ax, title in zip(axes, ['D_box  vs noise', 'D_wavelet (approx.)  vs noise']):
        ax.set_title(title)
        ax.set_xlabel('Noise σ')
        ax.legend()
    axes[0].set_ylabel('D_box')
    axes[1].set_ylabel('D_wavelet_approx')

    fig.suptitle('Noise robustness of fractal measures', fontsize=13)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, 'noise_robustness.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Завдання 3.10 — Фрактальна розмірність зображень (box-counting + wavelet)."
    )
    p.add_argument("--size",        default=512,  type=int,   help="Розмір зображення в пікселях (default: 512)")
    p.add_argument("--noise-sigma", default=None, type=float,
                   help="Єдиний рівень шуму для тесту замість стандартного набору (0.0–0.5)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 62)
    print("© 2026 Команда 10 — Фрактальні методи аналізу")
    print("Завдання 3.10: Фрактальна розмірність зображень")
    print(f"  Розмір зображень: {args.size}×{args.size} px")
    if args.noise_sigma is not None:
        print(f"  Рівень шуму: σ = {args.noise_sigma}")
    print("=" * 62)

    ensure_dir(DATA_DIR)
    ensure_dir(PLOT_DIR)

    # Generate images
    print("\nGenerating images...")
    np.random.seed(42)
    sz = args.size
    images = {
        'sierpinski': make_sierpinski(size=sz),
        'texture':    make_natural_texture(size=sz),
        'geometric':  make_geometric(size=sz),
    }
    for name, img in images.items():
        save_image_png(img, os.path.join(DATA_DIR, f'{name}.png'))

    # Part A — Box-counting
    print("\n--- Part A: Box-counting dimension ---")
    box_results = {}
    for name, img in images.items():
        D_box, sizes, counts = box_counting_dimension(img)
        box_results[name]    = D_box
        plot_boxcount_loglog(sizes, counts, D_box, name)
        print(f"  {name:<14}  D_box = {D_box:.4f}")

    # Part B — Wavelet complexity
    print("\n--- Part B: Wavelet complexity (approximate) ---")
    wav_results = {}
    for name, img in images.items():
        D_wav, levels, energies = wavelet_complexity_dimension(img)
        wav_results[name]       = D_wav
        plot_wavelet_energy(levels, energies, D_wav, name)
        print(f"  {name:<14}  D_wavelet_approx = {D_wav:.4f}"
              "  [scale-energy heuristic, not exact D_f]")

    # Part C — Noise robustness
    print("\n--- Part C: Noise robustness ---")
    sigmas = [args.noise_sigma, 0.0] if args.noise_sigma is not None else NOISE_SIGMAS
    rob_df = robustness_analysis(images, sigmas=sigmas)
    plot_noise_robustness(rob_df)
    save_table_csv(rob_df.set_index(['Image', 'Sigma']), OUT_CSV)

    # Console interpretation
    print("\n" + "=" * 62)
    print("Results summary")
    print("=" * 62)

    base = rob_df[rob_df['Sigma'] == 0.0]
    print(f"\n  {'Image':<14} {'D_box':>8} {'D_wavelet_approx':>18}")
    print("  " + "-" * 42)
    for _, row in base.iterrows():
        print(f"  {row['Image']:<14} {row['D_box']:>8.4f} {row['D_wavelet_approx']:>18.4f}")

    most_complex_box = base.loc[base['D_box'].idxmax(), 'Image']
    most_complex_wav = base.loc[base['D_wavelet_approx'].idxmax(), 'Image']
    print(f"\n  Highest D_box:          {most_complex_box}")
    print(f"  Highest D_wavelet:      {most_complex_wav}")
    print(f"  Sierpinski theoretical: D = {np.log(3)/np.log(2):.4f}  (log3/log2)")

    # Stability: std of metric across noise levels (sigma > 0) averaged over images
    noisy    = rob_df[rob_df['Sigma'] > 0]
    box_stab = noisy.groupby('Image')['D_box'].std().mean()
    wav_stab = noisy.groupby('Image')['D_wavelet_approx'].std().mean()
    print(f"\n  Noise stability (mean σ across images and noise levels):")
    print(f"    D_box              σ = {box_stab:.4f}")
    print(f"    D_wavelet_approx   σ = {wav_stab:.4f}")
    more_stable = "Box-counting" if box_stab <= wav_stab else "Wavelet complexity estimate"
    print(f"  → {more_stable} is more stable under Gaussian noise.")

    print("""
  Note on D_wavelet:
    D_wavelet is derived from the slope β of log(E_j) vs log(2^j) via the
    heuristic  D ≈ (7 − β) / 2.  It characterises how fine-scale detail
    energy decays across wavelet levels.  It is NOT the Hausdorff or
    box-counting dimension and should not be compared directly to D_box.
    Use it as a relative complexity index within the same method.
""")
    print("=" * 62)
    print("[Task 3.10 complete]")
    print("=" * 62)


if __name__ == '__main__':
    main()
