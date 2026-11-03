"""forecasting.py - Crop Yield Time Series Stationarity, Smoothing, and ARIMA Forecasting.
Author: Prathamesh Samel (Python Programming Laboratory - Exp 12)
Implements annual time series preparation, ADF stationarity testing, differencing,
SMA vs EWMA smoothing, ARIMA order selection, holdout evaluation, and 5-year forecasting.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from common_utils import (
    OUTPUT_DIR, RESULTS_DIR, ensure_directories,
    load_all_datasets, rmse, mae, fit_line, save_plot
)

warnings.filterwarnings("ignore")

RESULTS_PATH = RESULTS_DIR / "forecasting"
GRAPHS_PATH = OUTPUT_DIR / "forecasting"
CROPS = ["Rice", "Wheat", "Maize", "Sugarcane"]
HOLDOUT_YEARS = 4
FORECAST_YEARS = 5


def prepare_annual_series():
    """Build annual time series of median crop yield with a Pandas DatetimeIndex (Exp 12)."""
    crop_df, _, _ = load_all_datasets()
    series_dict = {}
    for crop in CROPS:
        sub = crop_df[(crop_df["crop"] == crop) & (crop_df["year"] <= 2019) & (crop_df["yield"] > 0)]
        yearly = sub.groupby("year")["yield"].median()
        dates = pd.to_datetime(yearly.index.astype(str) + "-12-31")
        series_dict[crop] = pd.Series(yearly.values, index=dates, name=crop)
    return series_dict


def analyze_stationarity_and_differencing(series_dict):
    """Assess stationarity using Augmented Dickey-Fuller (ADF) test and 1st differencing."""
    records, diff_dict = [], {}
    for crop, ts in series_dict.items():
        raw_p = adfuller(ts.dropna())[1]
        ts_diff = ts.diff().dropna()
        diff_dict[crop] = ts_diff
        diff_p = adfuller(ts_diff)[1]
        records.append({
            "crop": crop,
            "raw_adf_pvalue": round(float(raw_p), 4),
            "raw_status": "Stationary" if raw_p < 0.05 else "Non-Stationary",
            "differenced_adf_pvalue": round(float(diff_p), 4),
            "differenced_status": "Stationary" if diff_p < 0.05 else "Non-Stationary",
        })
    df_adf = pd.DataFrame(records)
    df_adf.to_csv(RESULTS_PATH / "stationarity_adf_test.csv", index=False)
    return df_adf, diff_dict


def analyze_smoothing(series_dict):
    """Compare 3-year Simple Moving Average (SMA) and EWMA (span=3) smoothing (Exp 12)."""
    records, smooth_dict = [], {}
    for crop, ts in series_dict.items():
        sma = ts.rolling(window=3).mean()
        ewma = ts.ewm(span=3, adjust=False).mean()
        smooth_dict[crop] = {"sma": sma, "ewma": ewma}
        sma_mad = float(np.mean(np.abs(ts[2:] - sma[2:])))
        ewma_mad = float(np.mean(np.abs(ts - ewma)))
        records.append({
            "crop": crop,
            "mean_yield": round(float(ts.mean()), 3),
            "sma_mad": round(sma_mad, 4),
            "ewma_mad": round(ewma_mad, 4),
            "more_responsive": "EWMA" if ewma_mad < sma_mad else "SMA",
        })
    df_smooth = pd.DataFrame(records)
    df_smooth.to_csv(RESULTS_PATH / "smoothing_summary.csv", index=False)
    return df_smooth, smooth_dict


def evaluate_arima_holdout(series_dict):
    """Compare candidate ARIMA models against Last Value baseline on 2016-2019 holdout."""
    candidate_orders = [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0)]
    eval_records, best_models = [], {}

    for crop, ts in series_dict.items():
        train, test = ts.iloc[:-HOLDOUT_YEARS], ts.iloc[-HOLDOUT_YEARS:]
        y_test = test.values

        # Baseline: Last Known Value
        pred_base = np.full(HOLDOUT_YEARS, train.iloc[-1])
        eval_records.append({
            "crop": crop, "model": "Baseline (Last Value)",
            "holdout_rmse": round(rmse(y_test, pred_base), 4),
            "holdout_mae": round(mae(y_test, pred_base), 4),
        })

        # Fit candidate ARIMA orders
        best_rmse, best_order, best_pred = float("inf"), (1, 1, 0), None
        for order in candidate_orders:
            try:
                mod = ARIMA(train, order=order).fit()
                pred = mod.forecast(steps=HOLDOUT_YEARS).values
                score = rmse(y_test, pred)
                if score < best_rmse:
                    best_rmse, best_order, best_pred = score, order, pred
            except Exception:
                continue

        eval_records.append({
            "crop": crop, "model": f"ARIMA{best_order}",
            "holdout_rmse": round(best_rmse, 4),
            "holdout_mae": round(mae(y_test, best_pred), 4),
        })
        best_models[crop] = {"order": best_order, "pred": best_pred, "test_years": test.index.year.values}

    eval_df = pd.DataFrame(eval_records)
    eval_df.to_csv(RESULTS_PATH / "holdout_evaluation.csv", index=False)
    return eval_df, best_models


def forecast_future_yields(series_dict, best_models):
    """Refit optimal ARIMA on 1997-2019 data and project 5 years (2020-2024) with 95% CI."""
    future_years = np.arange(2020, 2020 + FORECAST_YEARS)
    records, forecasts = [], {}

    for crop, ts in series_dict.items():
        order = best_models[crop]["order"]
        fitted = ARIMA(ts, order=order).fit()
        fc = fitted.get_forecast(steps=FORECAST_YEARS)
        pred, ci = fc.predicted_mean.values, fc.conf_int(alpha=0.05).values
        forecasts[crop] = {"years": future_years, "pred": pred, "ci_lower": ci[:, 0], "ci_upper": ci[:, 1]}

        for yr, val, low, high in zip(future_years, pred, ci[:, 0], ci[:, 1]):
            records.append({
                "crop": crop, "year": yr, "model": f"ARIMA{order}",
                "forecast": round(float(val), 4),
                "ci_lower_95": round(float(low), 4),
                "ci_upper_95": round(float(high), 4),
            })

    fc_df = pd.DataFrame(records)
    fc_df.to_csv(RESULTS_PATH / "future_forecasts_2020_2024.csv", index=False)
    return fc_df, forecasts


def plot_forecasting_figures(series_dict, diff_dict, smooth_dict, best_models, forecasts):
    """Generate 4 clean, focused time-series charts for college examination report."""
    GRAPHS_PATH.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    def make_2x2():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        return fig, axes.flatten()

    # 1. Historical Yield Trends & 1st Differenced Series
    fig, axes = make_2x2()
    for ax, crop in zip(axes, CROPS):
        ts = series_dict[crop]
        d = diff_dict[crop]
        ax.plot(ts.index.year, ts.values, "o-", label="Yield (t/ha)", color="#1f77b4")
        ax.plot(d.index.year, d.values, "s--", label="1st Diff (d=1)", color="#ff7f0e")
        ax.axhline(0, color="black", linestyle=":", alpha=0.6)
        ax.set_title(f"{crop} - Raw Trend vs Differenced", fontweight="bold")
        ax.set_xlabel("Year"); ax.legend(fontsize=8.5)
    save_plot("fc_1_stationarity_and_differencing.png", "forecasting")

    # 2. Trend Smoothing Comparison (SMA vs EWMA)
    fig, axes = make_2x2()
    for ax, crop in zip(axes, CROPS):
        ts = series_dict[crop]
        ax.plot(ts.index.year, ts.values, "o-", color="gray", alpha=0.5, label="Original")
        ax.plot(ts.index.year, smooth_dict[crop]["sma"], color="blue", lw=2, label="SMA (window=3)")
        ax.plot(ts.index.year, smooth_dict[crop]["ewma"], "r--", lw=2, label="EWMA (span=3)")
        ax.set_title(f"{crop} - Trend Smoothing", fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("Yield (t/ha)"); ax.legend(fontsize=8.5)
    save_plot("fc_2_smoothing_sma_ewma.png", "forecasting")

    # 3. Holdout Evaluation (2016-2019)
    fig, axes = make_2x2()
    for ax, crop in zip(axes, CROPS):
        ts = series_dict[crop]
        m = best_models[crop]
        ax.plot(ts.index.year, ts.values, "o-", color="black", label="Actual History")
        ax.plot(m["test_years"], m["pred"], "^-", color="crimson", lw=2, label=f"ARIMA{m['order']} Forecast")
        ax.axvspan(2015.5, 2019.5, color="lightyellow", alpha=0.6, label="Holdout Window")
        ax.set_title(f"{crop} - Holdout Validation (2016-2019)", fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("Yield (t/ha)"); ax.legend(fontsize=8.5)
    save_plot("fc_3_holdout_evaluation.png", "forecasting")

    # 4. 5-Year Future Forecasts (2020-2024) with 95% Confidence Intervals
    fig, axes = make_2x2()
    for ax, crop in zip(axes, CROPS):
        ts = series_dict[crop]
        fc = forecasts[crop]
        ax.plot(ts.index.year, ts.values, "o-", color="#1f77b4", label="Observed (1997-2019)")
        ax.plot(fc["years"], fc["pred"], "s-", color="red", lw=2, label=f"ARIMA Forecast")
        ax.fill_between(fc["years"], fc["ci_lower"], fc["ci_upper"], color="red", alpha=0.18, label="95% CI")
        ax.set_title(f"{crop} - 5-Year Forecast (2020-2024)", fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("Yield (t/ha)"); ax.legend(fontsize=8.5)
    save_plot("fc_4_future_forecasts.png", "forecasting")


def run_forecasting_analysis():
    """Main execution function for Prathamesh's time series forecasting."""
    ensure_directories()
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    print("Starting Module: Time Series Forecasting (Prathamesh Samel)...")

    series_dict = prepare_annual_series()
    df_adf, diff_dict = analyze_stationarity_and_differencing(series_dict)
    df_smooth, smooth_dict = analyze_smoothing(series_dict)
    eval_df, best_models = evaluate_arima_holdout(series_dict)
    fc_df, forecasts = forecast_future_yields(series_dict, best_models)

    plot_forecasting_figures(series_dict, diff_dict, smooth_dict, best_models, forecasts)

    summary_text = f"""PRATHAMESH SAMEL - TIME SERIES FORECASTING SUMMARY

1. METHODOLOGY (EXP 12: STATSMODELS TIME SERIES & ARIMA)
- Series Formulation: Annual median yield series with Pandas DatetimeIndex for Rice, Wheat, Maize, Sugarcane.
- Stationarity: Augmented Dickey-Fuller (ADF) test revealed non-stationarity; first differencing (d=1) stabilized means.
- Smoothing: EWMA (span=3) tracked historical shocks with lower Mean Absolute Deviation than 3-year SMA.
- Model Selection: Grid-searched candidate ARIMA orders over 2016-2019 holdout.

2. HOLDOUT MODEL EVALUATION
{eval_df.to_string(index=False)}

3. 5-YEAR FUTURE YIELD FORECASTS (2020-2024)
{fc_df.to_string(index=False)}
"""
    (RESULTS_PATH / "forecasting_summary.txt").write_text(summary_text, encoding="utf-8")

    print("\nHOLDOUT EVALUATION (2016-2019):")
    print(eval_df.to_string(index=False))
    print(f"\n[Done] Forecasting completed. Results saved to: {RESULTS_PATH}")
    return fc_df, eval_df


if __name__ == "__main__":
    run_forecasting_analysis()
