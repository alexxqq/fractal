"""
task2_fractal_ts.py — Fractal time series analysis (Team 10, Practical 2)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from utils import (
    ensure_dir, error_metrics, save_table_csv, plot_loglog,
    compute_hurst_rs, compute_mfdfa, classify_noise,
)

PROCESSED_CSV = os.path.join(_PROJECT, 'data', 'processed', 'eurusd_processed.csv')
PLOT_DIR      = os.path.join(_PROJECT, 'plots', 'task2')
DATA_DIR      = os.path.join(_PROJECT, 'data')

TRAIN_RATIO = 0.8


# ── Data loading ──────────────────────────────────────────────────────────────

def load_processed_data():
    """Load the processed EUR/USD data produced by Task 1."""
    df = pd.read_csv(PROCESSED_CSV, index_col=0, parse_dates=True)
    df = df.dropna(subset=['log_price', 'log_return'])
    return df


def train_test_split(series: pd.Series, ratio: float = TRAIN_RATIO):
    cut = int(len(series) * ratio)
    return series.iloc[:cut].copy(), series.iloc[cut:].copy()



def fracdiff(series: pd.Series, d: float, K: int = 50) -> pd.Series:
    """
    δ_0 = 1
    δ_k = δ_{k-1} * (k - 1 - d) / k   for k = 1, …, K
    """
    x      = np.asarray(series, dtype=float)
    N      = len(x)
    delta  = np.empty(K + 1)
    delta[0] = 1.0
    for k in range(1, K + 1):
        delta[k] = delta[k - 1] * (k - 1 - d) / k

    out = np.full(N, np.nan)
    for t in range(K, N):
        seg = x[t - K : t + 1][::-1]   # length K+1: x[t], x[t-1], …, x[t-K]
        out[t] = float(np.dot(delta, seg))

    return pd.Series(out, index=series.index, name=series.name)


# ── Part 1: R/S Hurst ─────────────────────────────────────────────────────────

def part1_hurst(log_return: pd.Series) -> float:
    """Compute Hurst exponent and save log-log R/S plot."""
    ensure_dir(PLOT_DIR)

    H, ns_arr, rs_arr = compute_hurst_rs(log_return.dropna().values)

    slope = plot_loglog(
        x        = ns_arr,
        y        = rs_arr,
        label    = 'R/S (EUR/USD log-return)',
        savepath = os.path.join(PLOT_DIR, 'rs_hurst.png'),
        xlabel   = 'log₁₀(n)',
        ylabel   = 'log₁₀(R/S)',
        title    = f'R/S Analysis  —  H = {H:.4f}',
    )

    print(f"\n[Task 2 | Part 1]  Hurst exponent H = {H:.4f}")
    print(f"  Classification: {classify_noise(H)}")
    return H


# ── Part 2: ARFIMA ────────────────────────────────────────────────────────────

def part2_arfima(log_return: pd.Series, H: float):

    d = H - 0.5
    print(f"\n[Task 2 | Part 2]  ARFIMA  d = {d:.4f}")

    # Fractional differencing on full series
    w_full = fracdiff(log_return, d, K=50)

    # Drop NaN warm-up period (first 50 obs)
    w_valid   = w_full.dropna()
    r_aligned = log_return.loc[w_valid.index]   # aligned actual log-returns

    # Train / test split on valid portion
    cut      = int(len(w_valid) * TRAIN_RATIO)
    w_train  = w_valid.iloc[:cut].reset_index(drop=True)
    w_test   = w_valid.iloc[cut:]
    r_test   = r_aligned.iloc[cut:]
    n_test   = len(w_test)

    # Fit ARMA(1,1) on fractionally differenced train
    arma_res = ARIMA(w_train, order=(1, 0, 1)).fit()
    print(f"  ARMA(1,1) on w_t  —  AIC = {arma_res.aic:.2f}")

    # One-step forecast (walk-forward would be ideal; static for simplicity)
    fc = arma_res.forecast(steps=n_test)
    fc = np.asarray(fc)

    # w_t = Δ^d r_t  so forecasted w_t ≈ forecasted r_t (long-memory removed)
    r_pred = fc
    r_actual = r_test.values

    metrics = error_metrics(r_actual, r_pred)
    print(f"  ARFIMA metrics  →  RMSE={metrics['RMSE']:.6f}  Theil_U={metrics['Theil_U']:.4f}")

    # ── save plot ──────────────────────────────────────────────────────────────
    ensure_dir(PLOT_DIR)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(np.arange(n_test), r_actual, label='Actual log-return',  linewidth=0.8)
    ax.plot(np.arange(n_test), r_pred,   label='ARFIMA forecast',    linewidth=0.8, alpha=0.8)
    ax.set_title(f'ARFIMA(1,{d:.3f},1) Forecast vs Actual  —  RMSE={metrics["RMSE"]:.6f}')
    ax.set_xlabel('Test step')
    ax.set_ylabel('Log-return')
    ax.legend()
    fig.tight_layout()
    savepath = os.path.join(PLOT_DIR, 'arfima_forecast.png')
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {savepath}")

    return metrics, d, arma_res


# ── Part 3: MF-DFA ────────────────────────────────────────────────────────────

def part3_mfdfa(log_return: pd.Series, q_list=None):
    """Run MF-DFA and save three plots: h(q), τ(q), multifractal spectrum f(α)."""
    if q_list is None:
        q_list = [-3, -2, -1, 0, 1, 2, 3]
    ensure_dir(PLOT_DIR)

    x      = log_return.dropna().values
    result = compute_mfdfa(x, q_list=q_list)

    q_arr   = result['q']
    h_q     = result['h_q']
    tau_q   = result['tau_q']
    alpha_q = result['alpha_q']
    f_alpha = result['f_alpha']

    # ── h(q) plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = ~np.isnan(h_q)
    ax.plot(q_arr[mask], h_q[mask], 'o-', linewidth=1.5)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, label='H = 0.5 (random)')
    ax.set_title('Generalised Hurst Exponent  h(q)')
    ax.set_xlabel('q')
    ax.set_ylabel('h(q)')
    ax.legend()
    fig.tight_layout()
    p = os.path.join(PLOT_DIR, 'hq_plot.png')
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  Saved: {p}")

    # ── τ(q) plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = ~np.isnan(tau_q)
    ax.plot(q_arr[mask], tau_q[mask], 's-', linewidth=1.5, color='teal')
    ax.set_title('Mass Exponent  τ(q)')
    ax.set_xlabel('q')
    ax.set_ylabel('τ(q)')
    fig.tight_layout()
    p = os.path.join(PLOT_DIR, 'tauq_plot.png')
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  Saved: {p}")

    # ── f(α) spectrum ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = ~(np.isnan(alpha_q) | np.isnan(f_alpha))
    ax.plot(alpha_q[mask], f_alpha[mask], 'D-', linewidth=1.5, color='crimson')
    ax.set_title('Multifractal Spectrum  f(α)')
    ax.set_xlabel('α  (Hölder exponent)')
    ax.set_ylabel('f(α)')
    fig.tight_layout()
    p = os.path.join(PLOT_DIR, 'falpha_plot.png')
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  Saved: {p}")

    # ── console interpretation ─────────────────────────────────────────────────
    h_q_valid = h_q[~np.isnan(h_q)]
    delta_h   = float(np.ptp(h_q_valid)) if len(h_q_valid) > 1 else float('nan')
    print(f"\n[Task 2 | Part 3]  MF-DFA results")
    print(f"  h(q) range: [{h_q_valid.min():.4f}, {h_q_valid.max():.4f}]  Δh = {delta_h:.4f}")
    if delta_h > 0.1:
        print(f"  → Multifractal: Δh = {delta_h:.4f} > 0.1  (heterogeneous scaling)")
    else:
        print(f"  → Near-monofractal: Δh = {delta_h:.4f} ≤ 0.1")

    return result


# ── Comparison table ──────────────────────────────────────────────────────────

def save_comparison(arfima_metrics: dict, d: float, H: float):
    """
    Load Task 1 metrics CSV and append ARFIMA row for a side-by-side comparison.
    Saves to data/metrics_task2.csv.
    """
    task1_csv = os.path.join(_PROJECT, 'plots', 'task1', 'metrics_task1.csv')
    rows = []

    if os.path.exists(task1_csv):
        t1 = pd.read_csv(task1_csv, index_col=0)
        for idx in t1.index:
            rows.append({'Model': idx, **{c: t1.loc[idx, c] for c in t1.columns}})

    rows.append({
        'Model'  : f'ARFIMA(1,{d:.3f},1)',
        'MSE'    : arfima_metrics['MSE'],
        'RMSE'   : arfima_metrics['RMSE'],
        'MAD'    : arfima_metrics['MAD'],
        'MAPE'   : arfima_metrics['MAPE'],
        'Theil_U': arfima_metrics['Theil_U'],
    })

    df = pd.DataFrame(rows).set_index('Model')
    save_table_csv(df, os.path.join(DATA_DIR, 'metrics_task2.csv'))

    print("\n[Task 2]  Model comparison (log-return scale):")
    print(df.to_string())


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Завдання 2 — ARFIMA та MF-DFA фрактальний аналіз часового ряду."
    )
    p.add_argument("--q-min",       default=-3, type=int,   help="Мінімальний момент q для MF-DFA (default: -3)")
    p.add_argument("--q-max",       default=3,  type=int,   help="Максимальний момент q для MF-DFA (default: 3)")
    p.add_argument("--train-ratio", default=0.8, type=float, help="Частка навчальної вибірки (default: 0.8)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 62)
    print("© 2026 Команда 10 — Фрактальні методи аналізу")
    print("Завдання 2: ARFIMA + MF-DFA фрактальний аналіз")
    print(f"  q ∈ [{args.q_min}, {args.q_max}]  |  train ratio: {args.train_ratio:.0%}")
    print("=" * 62)

    df         = load_processed_data()
    log_return = df['log_return'].dropna()

    print(f"  Series length: {len(log_return)} observations")

    # Part 1 — Hurst
    H = part1_hurst(log_return)

    # Part 2 — ARFIMA
    arfima_metrics, d, _ = part2_arfima(log_return, H)

    # Part 3 — MF-DFA
    q_list = list(range(args.q_min, args.q_max + 1))
    part3_mfdfa(log_return, q_list=q_list)

    # Summary comparison table
    save_comparison(arfima_metrics, d, H)

    print("\n" + "=" * 60)
    print("[Task 2 complete]")
    print(f"  H = {H:.4f}  ({classify_noise(H)})")
    print(f"  d = H - 0.5 = {d:.4f}  (fractional integration order)")
    print(f"  ARFIMA RMSE = {arfima_metrics['RMSE']:.6f}  "
          f"Theil_U = {arfima_metrics['Theil_U']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
