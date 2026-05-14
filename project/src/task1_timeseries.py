"""
task1_timeseries.py — ARMA(1,1) and ARIMA(2,1,2) on EUR/USD data.
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import ensure_dir, error_metrics, save_table_csv, plot_series

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ── Path constants ─────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(_HERE)
RAW_DIR   = os.path.join(ROOT, 'data', 'raw')
PROC_DIR  = os.path.join(ROOT, 'data', 'processed')
PLOTS_DIR = os.path.join(ROOT, 'plots', 'task1')


# ── 1. Data acquisition ────────────────────────────────────────────────────

def _download_yfinance(ticker: str = "EURUSD=X",
                       start: str  = "2015-01-01",
                       end: str    = "2024-12-31") -> pd.DataFrame:
    import yfinance as yf
    print(f"  Downloading {ticker} from yfinance ({start} → {end})...")
    df = yf.download(ticker, start=start, end=end,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError("yfinance returned empty DataFrame")
    # Flatten MultiIndex columns produced by some yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Close']].copy()
    df.index = pd.to_datetime(df.index)
    print(f"  Downloaded {len(df)} rows.")
    return df


def _synthetic_fallback() -> pd.DataFrame:
    """
    Synthetic EUR/USD-like log-price series.
    Built as a random walk driven by AR(1) noise (phi=0.25) so the
    log-returns are mildly autocorrelated — realistic for FX data.
    """
    print("  Using synthetic fallback (AR-noise random walk).")
    np.random.seed(42)
    n     = 2500
    dates = pd.date_range(start='2015-01-01', periods=n, freq='B')
    phi   = 0.25
    innov = np.random.normal(0, 0.004, n)
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = phi * noise[t - 1] + innov[t]
    log_p  = np.cumsum(noise)
    prices = np.exp(log_p + np.log(1.10))   # start near EUR/USD 1.10
    return pd.DataFrame({'Close': prices}, index=dates)


def load_raw_data(ticker: str = "EURUSD=X",
                  start: str  = "2015-01-01",
                  end: str    = "2024-12-31") -> pd.DataFrame:
    try:
        return _download_yfinance(ticker, start, end)
    except Exception as exc:
        print(f"  yfinance failed: {exc}")
        return _synthetic_fallback()


# ── 2. Preprocessing ───────────────────────────────────────────────────────

def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Add log_price = ln(Close) and log_return = diff(log_price).
    Drop the single leading NaN from diff().
    """
    df               = raw.copy()
    df['log_price']  = np.log(df['Close'])
    df['log_return'] = df['log_price'].diff()
    return df.dropna()


def train_test_split(series: pd.Series, ratio: float = 0.8):
    n = int(len(series) * ratio)
    return series.iloc[:n].copy(), series.iloc[n:].copy()


# ── 3. Diagnostics helpers ─────────────────────────────────────────────────

def run_adf(series: pd.Series, name: str) -> dict:
    """Augmented Dickey-Fuller test. Returns a flat result dict."""
    res = adfuller(series.dropna(), autolag='AIC')
    return {
        'series':           name,
        'ADF_statistic':    round(res[0], 6),
        'p_value':          round(res[1], 6),
        'critical_1pct':    round(res[4]['1%'], 6),
        'critical_5pct':    round(res[4]['5%'], 6),
        'stationary_5pct':  res[1] < 0.05,
    }


def run_ljungbox(residuals: pd.Series, lags: int = 10) -> dict:
    """Ljung-Box portmanteau test. Returns summary dict."""
    lb    = acorr_ljungbox(residuals.dropna(), lags=lags, return_df=True)
    min_p = float(lb['lb_pvalue'].min())
    return {
        'lags_tested':          lags,
        'min_p_value':          round(min_p, 6),
        'residuals_white_noise': min_p > 0.05,
    }


# ── 4. Plotting ────────────────────────────────────────────────────────────

def plot_log_price_and_return(df: pd.DataFrame, savepath: str) -> None:
    ensure_dir(os.path.dirname(savepath))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    ax1.plot(df.index, df['log_price'], linewidth=0.7, color='steelblue')
    ax1.set_title('EUR/USD Log-Price  [ARIMA(2,1,2) input]')
    ax1.set_ylabel('ln(Close)')

    ax2.plot(df.index, df['log_return'], linewidth=0.6, color='darkorange')
    ax2.axhline(0, color='gray', linewidth=0.6, linestyle='--')
    ax2.set_title('EUR/USD Log-Return  [ARMA(1,1) input]')
    ax2.set_ylabel('Δ ln(Close)')
    ax2.set_xlabel('Date')

    fig.suptitle('EUR/USD Daily Data — Preprocessing', fontsize=13)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {savepath}")


def plot_acf_pacf_figure(series: pd.Series, lags: int, savepath: str) -> None:
    ensure_dir(os.path.dirname(savepath))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf( series.dropna(), lags=lags, ax=ax1, title='ACF  — Log-Return')
    plot_pacf(series.dropna(), lags=lags, ax=ax2, title='PACF — Log-Return',
              method='ywm')
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {savepath}")


def plot_forecast_figure(
    train: pd.Series,
    test:  pd.Series,
    fc:    pd.Series,
    title: str,
    ylabel: str,
    savepath: str,
    context_n: int = 200,
) -> None:
    """Plot last *context_n* train points + full test actuals + forecast."""
    ensure_dir(os.path.dirname(savepath))
    ctx = train.iloc[-context_n:]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(ctx.index,  ctx.values,  color='steelblue', linewidth=0.8,
            label=f'Train (last {context_n})')
    ax.plot(test.index, test.values, color='black',     linewidth=0.9,
            label='Actual (test)')
    ax.plot(fc.index,   fc.values,   color='crimson',   linewidth=1.1,
            linestyle='--', label='Forecast')
    ax.axvline(x=test.index[0], color='gray', linestyle=':', linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {savepath}")


def plot_residuals_figure(
    residuals:  pd.Series,
    model_name: str,
    savepath:   str,
) -> None:
    """Residual line plot + histogram, side-by-side."""
    ensure_dir(os.path.dirname(savepath))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(residuals.index, residuals.values, linewidth=0.6, color='steelblue')
    ax1.axhline(0, color='red', linewidth=0.8, linestyle='--')
    ax1.set_title(f'{model_name} — Residuals over time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Residual')

    ax2.hist(residuals.dropna().values, bins=50,
             color='steelblue', edgecolor='white', linewidth=0.3)
    ax2.set_title(f'{model_name} — Residual distribution')
    ax2.set_xlabel('Residual value')
    ax2.set_ylabel('Frequency')

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {savepath}")


# ── 5. Model fitting ───────────────────────────────────────────────────────

def fit_arma_11(train_returns: pd.Series):
    """
    ARMA(1,1) on log-returns.
    Uses reset_index(drop=True) to give statsmodels a plain integer index;
    this silences the 'no associated frequency' warning while keeping named
    parameters (ar.L1, ma.L1, sigma2) in the fitted result.
    """
    model  = ARIMA(train_returns.reset_index(drop=True), order=(1, 0, 1))
    result = model.fit()
    return result


def fit_arima_212(train_log_price: pd.Series):
    """
    ARIMA(2,1,2) on log-prices.
    d=1 differencing applied internally.  Integer index avoids frequency warnings.
    """
    model  = ARIMA(train_log_price.reset_index(drop=True), order=(2, 1, 2))
    result = model.fit()
    return result


# ── 6. Forecasting ─────────────────────────────────────────────────────────

def forecast_arma(result, test_returns: pd.Series) -> pd.Series:
    """
    Multi-step ahead forecast of log-returns.
    Returns a Series indexed to match test_returns.
    """
    fc_values = np.array(result.forecast(steps=len(test_returns)))
    return pd.Series(fc_values, index=test_returns.index, name='arma_forecast')


def forecast_arima_as_returns(
    result,
    test_log_price:  pd.Series,
    train_log_price: pd.Series,
) -> tuple:
    """
    Forecast log-prices with ARIMA, then convert to log-return scale.

    Conversion:
        predicted_return_t = predicted_log_price_t - actual_log_price_{t-1}

    'Actual previous' for the first test step is the last training value.
    For subsequent steps, actual (not forecasted) previous values are used.
    This keeps the comparison to ARMA(1,1) on the same return scale.

    Returns
    -------
    fc_returns   : pd.Series  (log-return scale, same index as test_log_price)
    fc_log_price : pd.Series  (raw log-price forecasts)
    """
    steps        = len(test_log_price)
    fc_lp_values = np.array(result.forecast(steps=steps))
    fc_log_price = pd.Series(fc_lp_values, index=test_log_price.index,
                             name='arima_fc_log_price')

    # Previous actual log-price: [last_train] + test[0:-1]
    prev_actual = pd.concat([
        train_log_price.iloc[[-1]],
        test_log_price.iloc[:-1],
    ])
    prev_actual.index = test_log_price.index

    fc_returns = pd.Series(
        fc_lp_values - prev_actual.values,
        index=test_log_price.index,
        name='arima_fc_log_return',
    )
    return fc_returns, fc_log_price


# ── 7. Parameter extraction ────────────────────────────────────────────────

def extract_params(result, model_name: str) -> pd.DataFrame:
    params = pd.DataFrame({
        'Model':     model_name,
        'Parameter': result.params.index,
        'Estimate':  result.params.values.round(8),
        'Std_Error': result.bse.values.round(8),
        'p_value':   result.pvalues.values.round(6),
    })
    return params.reset_index(drop=True)


# ── 8. Console interpretation ──────────────────────────────────────────────

def print_interpretation(
    adf_logprice:  dict,
    adf_logreturn: dict,
    metrics_arma:  dict,
    metrics_arima: dict,
    lb_arma:       dict,
    lb_arima:      dict,
) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("TASK 1 — INTERPRETATION")
    print(sep)

    # Stationarity
    if adf_logreturn['stationary_5pct']:
        print("[OK] Log-return is STATIONARY (ADF p={p_value:.4f}) "
              "-> valid input for ARMA(1,1)".format(**adf_logreturn))
    else:
        print("[!!] Log-return is NOT stationary at 5% — check preprocessing")

    if not adf_logprice['stationary_5pct']:
        print("[OK] Log-price is NON-STATIONARY (ADF p={p_value:.4f}) "
              "-> ARIMA d=1 differencing is appropriate".format(**adf_logprice))
    else:
        print("[!!] Log-price appears stationary — reconsider ARIMA differencing order")

    # Forecast comparison (log-return scale)
    print()
    arma_rmse  = metrics_arma.get('RMSE',   float('inf'))
    arima_rmse = metrics_arima.get('RMSE',  float('inf'))
    arma_mape  = metrics_arma.get('MAPE',   float('inf'))
    arima_mape = metrics_arima.get('MAPE',  float('inf'))

    better_rmse = 'ARMA(1,1)' if arma_rmse <= arima_rmse else 'ARIMA(2,1,2)'
    better_mape = 'ARMA(1,1)' if arma_mape <= arima_mape else 'ARIMA(2,1,2)'

    print(f"RMSE  — ARMA(1,1): {arma_rmse:.8f}  |  ARIMA(2,1,2): {arima_rmse:.8f}"
          f"  -> lower: {better_rmse}")
    print(f"MAPE  — ARMA(1,1): {arma_mape:.4f}%  |  ARIMA(2,1,2): {arima_mape:.4f}%"
          f"  -> lower: {better_mape}")
    if arma_mape > 50 or arima_mape > 50:
        print("  [NOTE] MAPE is unreliable for log-returns: values near zero inflate"
              " relative errors arbitrarily. Prefer RMSE and Theil_U for this series.")

    # Residuals
    print()
    for name, lb in [('ARMA(1,1)', lb_arma), ('ARIMA(2,1,2)', lb_arima)]:
        if lb['residuals_white_noise']:
            print(f"[OK] {name} residuals: Ljung-Box min p={lb['min_p_value']:.4f} "
                  f"-> no significant autocorrelation")
        else:
            print(f"[!!] {name} residuals: Ljung-Box min p={lb['min_p_value']:.4f} "
                  f"-> residual autocorrelation detected")

    print(sep)


# ── main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Завдання 1 — ARMA(1,1) та ARIMA(2,1,2) на даних валютної пари."
    )
    p.add_argument("--ticker",     default="EURUSD=X",   help="Тікер yfinance (default: EURUSD=X)")
    p.add_argument("--start",      default="2015-01-01", help="Дата початку (YYYY-MM-DD)")
    p.add_argument("--end",        default="2024-12-31", help="Дата кінця   (YYYY-MM-DD)")
    p.add_argument("--test-ratio", default=0.2, type=float,
                   help="Частка тестової вибірки (default: 0.2)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 62)
    print("© 2026 Команда 10 — Фрактальні методи аналізу")
    print("Завдання 1: ARMA(1,1) та ARIMA(2,1,2)")
    print(f"  Тікер: {args.ticker}  |  {args.start} → {args.end}")
    print(f"  Тестова вибірка: {int(args.test_ratio * 100)}%")
    print("=" * 62)

    for d in (RAW_DIR, PROC_DIR, PLOTS_DIR):
        ensure_dir(d)

    # ── Step 1: Load and save raw data ─────────────────────────────────────
    print("\n[1/6] Loading data...")
    raw = load_raw_data(args.ticker, args.start, args.end)
    raw_path = os.path.join(RAW_DIR, 'eurusd_daily.csv')
    raw.to_csv(raw_path)
    print(f"  Saved: {raw_path}")

    # ── Step 2: Preprocess ─────────────────────────────────────────────────
    print("\n[2/6] Preprocessing...")
    df = preprocess(raw)
    proc_path = os.path.join(PROC_DIR, 'eurusd_processed.csv')
    df.to_csv(proc_path)
    print(f"  Saved: {proc_path}")

    # ── Step 3: Visualize series ───────────────────────────────────────────
    print("\n[3/6] Visualising series...")
    plot_log_price_and_return(df, os.path.join(PLOTS_DIR, 'series_log_price_return.png'))

    # ── Step 4: Stationarity and ACF/PACF ─────────────────────────────────
    print("\n[4/6] Stationarity tests and ACF/PACF...")
    adf_lp = run_adf(df['log_price'],  'log_price')
    adf_lr = run_adf(df['log_return'], 'log_return')
    diag_df = pd.DataFrame([adf_lp, adf_lr]).set_index('series')
    save_table_csv(diag_df, os.path.join(PLOTS_DIR, 'diagnostics_task1.csv'))

    plot_acf_pacf_figure(
        df['log_return'], lags=40,
        savepath=os.path.join(PLOTS_DIR, 'acf_pacf_log_return.png'),
    )

    # ── Step 5: Split ──────────────────────────────────────────────────────
    log_return = df['log_return'].dropna()
    log_price  = df['log_price']

    train_return, test_return = train_test_split(log_return, 1 - args.test_ratio)
    train_price,  test_price  = train_test_split(log_price,  1 - args.test_ratio)

    print(f"\n  Train: {len(train_return)} obs  |  Test: {len(test_return)} obs")

    # ── Step 6: Fit models ─────────────────────────────────────────────────
    print("\n[5/6] Fitting models and forecasting...")

    print("  ARMA(1,1) on log-returns...")
    arma_result = fit_arma_11(train_return)
    arma_fc     = forecast_arma(arma_result, test_return)

    print("  ARIMA(2,1,2) on log-prices...")
    arima_result                = fit_arima_212(train_price)
    arima_fc_returns, _ = forecast_arima_as_returns(
        arima_result, test_price, train_price,
    )

    # ── Step 7: Residual diagnostics ───────────────────────────────────────
    print("\n[6/6] Diagnostics, plots, and metrics...")

    # Re-attach DatetimeIndex — models were fitted with reset_index(drop=True)
    # so residuals carry integer indices; re-assign to training dates for plotting.
    arma_resid  = pd.Series(arma_result.resid.values,  index=train_return.index)
    arima_resid = pd.Series(arima_result.resid.values, index=train_price.index)

    plot_residuals_figure(arma_resid,  'ARMA(1,1)',    os.path.join(PLOTS_DIR, 'residuals_arma.png'))
    plot_residuals_figure(arima_resid, 'ARIMA(2,1,2)', os.path.join(PLOTS_DIR, 'residuals_arima.png'))

    lb_arma  = run_ljungbox(arma_resid)
    lb_arima = run_ljungbox(arima_resid)

    # ── Step 8: Forecast plots ─────────────────────────────────────────────
    plot_forecast_figure(
        train=train_return, test=test_return, fc=arma_fc,
        title='ARMA(1,1) — Forecast vs. Actual (Log-Return scale)',
        ylabel='Log-Return',
        savepath=os.path.join(PLOTS_DIR, 'arma_forecast.png'),
    )
    plot_forecast_figure(
        train=train_return, test=test_return, fc=arima_fc_returns,
        title='ARIMA(2,1,2) — Forecast converted to Log-Return scale',
        ylabel='Log-Return',
        savepath=os.path.join(PLOTS_DIR, 'arima_forecast_returns_scale.png'),
    )

    # ── Step 9: Metrics ────────────────────────────────────────────────────
    metrics_arma  = error_metrics(test_return.values, arma_fc.values)
    metrics_arima = error_metrics(test_return.values, arima_fc_returns.values)

    metrics_df = pd.DataFrame([
        {'Model': 'ARMA(1,1)',    **metrics_arma},
        {'Model': 'ARIMA(2,1,2)', **metrics_arima},
    ]).set_index('Model')
    save_table_csv(metrics_df, os.path.join(PLOTS_DIR, 'metrics_task1.csv'))
    print("\nMetrics (log-return scale):")
    print(metrics_df.to_string())

    # ── Step 10: Parameter tables ──────────────────────────────────────────
    params_df = pd.concat([
        extract_params(arma_result,  'ARMA(1,1)'),
        extract_params(arima_result, 'ARIMA(2,1,2)'),
    ], ignore_index=True)
    save_table_csv(params_df, os.path.join(PLOTS_DIR, 'model_params_task1.csv'))

    # ── Step 11: Interpretation ────────────────────────────────────────────
    print_interpretation(adf_lp, adf_lr, metrics_arma, metrics_arima,
                         lb_arma, lb_arima)


if __name__ == '__main__':
    main()
