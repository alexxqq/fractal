"""
utils.py — shared helpers used across all task modules.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


# ── Metrics ────────────────────────────────────────────────────────────────

def error_metrics(actual, predicted) -> dict:

    actual    = np.asarray(actual,    dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()

    n = min(len(actual), len(predicted))
    actual, predicted = actual[:n], predicted[:n]

    valid = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[valid], predicted[valid]

    if len(actual) == 0:
        nan = float('nan')
        return dict(MSE=nan, RMSE=nan, MAD=nan, MAPE=nan, Theil_U=nan)

    errors = actual - predicted
    mse    = float(np.mean(errors ** 2))
    rmse   = float(np.sqrt(mse))
    mad    = float(np.mean(np.abs(errors)))

    # MAPE — only over non-zero actual values
    nz   = actual != 0
    mape = float(np.mean(np.abs(errors[nz] / actual[nz])) * 100) if nz.any() else float('nan')

    # Theil U — compare to naive random-walk: y_hat_t = y_{t-1}
    if len(actual) > 1:
        naive_rmse = float(np.sqrt(np.mean((actual[1:] - actual[:-1]) ** 2)))
        theil_u    = (rmse / naive_rmse) if naive_rmse > 0 else float('nan')
    else:
        theil_u = float('nan')

    return dict(
        MSE     = round(mse,     10),
        RMSE    = round(rmse,    10),
        MAD     = round(mad,     10),
        MAPE    = round(mape,     4) if not np.isnan(mape)    else float('nan'),
        Theil_U = round(theil_u,  6) if not np.isnan(theil_u) else float('nan'),
    )



def save_table_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path)
    print(f"  Saved: {path}")



def plot_series(
    series_dict: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    savepath: str,
    figsize: tuple = (13, 4),
) -> None:

    ensure_dir(os.path.dirname(savepath))
    fig, ax = plt.subplots(figsize=figsize)
    for label, data in series_dict.items():
        ax.plot(data, label=label, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(series_dict) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {savepath}")


# ── Fractal analysis ───────────────────────────────────────────────────────

def compute_hurst_rs(series, min_n: int = 10, max_n: int = None,
                     n_points: int = 20):

    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    N = len(x)

    max_n = min(max_n or N // 2, N // 2)
    min_n = max(min_n, 10)

    ns = np.unique(np.round(
        np.logspace(np.log10(min_n), np.log10(max_n), n_points)
    ).astype(int))

    rs_means, valid_ns = [], []

    for n in ns:
        num_segs = N // n
        if num_segs < 2:
            continue
        rs_vals = []
        for i in range(num_segs):
            seg = x[i * n : (i + 1) * n]
            Z   = np.cumsum(seg - seg.mean())
            R   = Z.max() - Z.min()
            S   = seg.std(ddof=1)
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            rs_means.append(np.mean(rs_vals))
            valid_ns.append(n)

    ns_arr = np.array(valid_ns, dtype=float)
    rs_arr = np.array(rs_means, dtype=float)

    coeffs = np.polyfit(np.log10(ns_arr), np.log10(rs_arr), 1)
    H      = float(coeffs[0])
    return H, ns_arr, rs_arr


def compute_mfdfa(series, q_list=None, scales=None, poly_deg: int = 1):

    if q_list is None:
        q_list = [-3, -2, -1, 0, 1, 2, 3]

    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    N = len(x)

    if scales is None:
        s_min  = 10
        s_max  = max(N // 4, s_min + 1)
        scales = np.unique(np.round(
            np.logspace(np.log10(s_min), np.log10(s_max), 20)
        ).astype(int))

    # 1. Profile
    Y = np.cumsum(x - x.mean())

    h_q = np.full(len(q_list), np.nan)

    for qi, q in enumerate(q_list):
        fq_scales = []

        for s in scales:
            s  = int(s)
            Ns = N // s
            if Ns < 2:
                fq_scales.append(np.nan)
                continue

            # 2. Detrended variances for 2*Ns segments
            F2 = []
            for direction in (1, -1):
                for v in range(Ns):
                    if direction == 1:
                        seg = Y[v * s : (v + 1) * s]
                    else:
                        seg = Y[N - (v + 1) * s : N - v * s]
                    t_seg  = np.arange(len(seg))
                    trend  = np.polyval(np.polyfit(t_seg, seg, poly_deg), t_seg)
                    resid  = seg - trend
                    f2_val = np.mean(resid ** 2)
                    if f2_val > 0:
                        F2.append(f2_val)

            if len(F2) < 2:
                fq_scales.append(np.nan)
                continue

            F2_arr = np.array(F2)

            # 3. Fluctuation function
            if q == 0:

                Fq = np.exp(0.5 * np.mean(np.log(F2_arr)))
            else:
                avg = np.mean(F2_arr ** (q / 2.0))
                Fq  = abs(avg) ** (1.0 / q) * np.sign(avg) if avg != 0 else np.nan

            fq_scales.append(Fq)

        # 4. Log-log regression
        valid = [(s, f) for s, f in zip(scales, fq_scales)
                 if f is not None and not np.isnan(f) and f > 0]
        if len(valid) < 3:
            continue
        vs, vf     = zip(*valid)
        log_s      = np.log(np.array(vs, dtype=float))
        log_f      = np.log(np.array(vf, dtype=float))
        h_q[qi]   = np.polyfit(log_s, log_f, 1)[0]

    q_arr  = np.array(q_list, dtype=float)
    tau_q  = q_arr * h_q - 1.0                    # 5. Mass exponent
    alpha_q = np.gradient(tau_q, q_arr)            # 6. Legendre
    f_alpha = q_arr * alpha_q - tau_q

    return dict(q=q_arr, h_q=h_q, tau_q=tau_q, alpha_q=alpha_q, f_alpha=f_alpha)


def classify_noise(H: float) -> str:
    if H < 0.5:
        return f'Anti-persistent  (pink noise,   H = {H:.3f} < 0.5)'
    if abs(H - 0.5) < 0.02:
        return f'Random walk      (white noise,  H = {H:.3f} ≈ 0.5)'
    if H < 0.75:
        return f'Persistent       (black noise,  H = {H:.3f})'
    return     f'Strongly persistent (black noise, H = {H:.3f} > 0.75)'


def plot_loglog(
    x,
    y,
    label: str,
    savepath: str,
    xlabel: str  = 'log(x)',
    ylabel: str  = 'log(y)',
    title: str   = 'Log-Log Plot',
    figsize: tuple = (7, 5),
) -> float:

    ensure_dir(os.path.dirname(savepath))
    lx = np.log10(np.asarray(x, dtype=float))
    ly = np.log10(np.asarray(y, dtype=float))

    coeffs   = np.polyfit(lx, ly, 1)
    slope    = coeffs[0]
    fit_line = np.poly1d(coeffs)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(lx, ly, s=30, label=label, zorder=3)
    ax.plot(lx, fit_line(lx), color='crimson', linewidth=1.5,
            label=f'Fit  slope = {slope:.4f}')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {savepath}")
    return float(slope)
