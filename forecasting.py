import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_utils import load_data, save_and_show, fit_line, rmse, mae, mape

crops = ["Rice", "Wheat", "Maize", "Sugarcane"]
methods = ["Last value", "Average yearly change", "Linear trend"]
last_year = 2019
holdout_years = 4
forecast_years = 5


def get_yearly_series(df, crop):
    data = df[(df["crop"] == crop) & (df["year"] <= last_year) & (df["yield"] > 0)]
    return data.groupby("year")["yield"].median()


def forecast(values, steps, method):
    result = np.zeros(steps)
    if method == "Last value":
        result = values[-1] * np.ones(steps)
    elif method == "Average yearly change":
        average_change = (values[-1] - values[0]) / (len(values) - 1)
        for h in range(1, steps + 1):
            result[h - 1] = values[-1] + h * average_change
    elif method == "Linear trend":
        slope, intercept = fit_line(np.arange(len(values)), values)
        for h in range(1, steps + 1):
            result[h - 1] = intercept + slope * (len(values) - 1 + h)
    return result


def trend_summary(all_series):
    rows = []
    for crop in crops:
        series = all_series[crop]
        slope, intercept = fit_line(series.index.to_numpy(), series.to_numpy())
        rows.append({
            "crop": crop,
            "first_year_value": round(series.iloc[0], 3),
            "last_year_value": round(series.iloc[-1], 3),
            "change_per_year": round(slope, 4),
            "growth_pct_per_year": round(slope / series.mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def evaluate_methods(all_series):
    rows = []
    for crop in crops:
        values = all_series[crop].to_numpy()
        train = values[:-holdout_years]
        actual = values[-holdout_years:]
        for method in methods:
            predicted = forecast(train, holdout_years, method)
            rows.append({
                "crop": crop,
                "method": method,
                "RMSE": round(rmse(actual, predicted), 4),
                "MAE": round(mae(actual, predicted), 4),
                "MAPE_pct": round(mape(actual, predicted), 2),
            })
    return pd.DataFrame(rows)


def best_method(errors, crop):
    crop_errors = errors[errors["crop"] == crop].sort_values("RMSE")
    return crop_errors["method"].iloc[0]


def future_forecast(series, method):
    values = series.to_numpy()
    predicted = forecast(values, forecast_years, method)
    years = np.arange(series.index[-1] + 1, series.index[-1] + forecast_years + 1)
    return pd.DataFrame({"year": years, "forecast_yield": np.round(predicted, 3)})


def plot_history(all_series):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    flat_axes = axes.flatten()
    for i in range(len(crops)):
        series = all_series[crops[i]]
        slope, intercept = fit_line(series.index.to_numpy(), series.to_numpy())
        flat_axes[i].plot(series.index, series.values, marker="o", label="Median yield")
        flat_axes[i].plot(series.index, intercept + slope * series.index, "r--", label="Linear trend")
        flat_axes[i].set_title(crops[i])
        flat_axes[i].set_xlabel("Year")
        flat_axes[i].set_ylabel("Median yield")
        flat_axes[i].legend()
    plt.suptitle("Yearly median yield, 1997-2019")
    save_and_show("fc_1_history_and_trend.png")


def plot_holdout(all_series):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    flat_axes = axes.flatten()
    for i in range(len(crops)):
        series = all_series[crops[i]]
        values = series.to_numpy()
        train = values[:-holdout_years]
        test_years = series.index[-holdout_years:]
        flat_axes[i].plot(series.index, values, color="black", marker="o", label="Actual")
        for method in methods:
            predicted = forecast(train, holdout_years, method)
            flat_axes[i].plot(test_years, predicted, marker="s", linestyle="--", label=method)
        flat_axes[i].set_title(crops[i])
        flat_axes[i].set_xlabel("Year")
        flat_axes[i].set_ylabel("Median yield")
        flat_axes[i].legend(fontsize=8)
    plt.suptitle("Forecasts for 2016-2019 made using data up to 2015")
    save_and_show("fc_2_holdout_comparison.png")


def plot_forecasts(all_series, all_forecasts, chosen):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    flat_axes = axes.flatten()
    for i in range(len(crops)):
        series = all_series[crops[i]]
        table = all_forecasts[crops[i]]
        flat_axes[i].plot(series.index, series.values, marker="o", label="Actual")
        flat_axes[i].plot(table["year"], table["forecast_yield"], "r-s", label="Forecast (" + chosen[crops[i]] + ")")
        flat_axes[i].set_title(crops[i])
        flat_axes[i].set_xlabel("Year")
        flat_axes[i].set_ylabel("Median yield")
        flat_axes[i].legend(fontsize=8)
    plt.suptitle("Yield forecast for 2020-2024")
    save_and_show("fc_3_forecasts.png")


def print_limitations():
    print("\nLIMITATIONS")
    print("1. There are only 23 yearly values per crop, so forecasts are simple trend extensions.")
    print("2. The best method is chosen using only 4 test years, so a close result is weak evidence.")
    print("3. No weather, price or policy data is used, so a drought year cannot be predicted.")
    print("4. The series is the median of all states, so it describes a typical state and not one state.")
    print("5. The year 2020 is left out because the file has only 37 rows for it.")


def main():
    df = load_data()

    all_series = {}
    for crop in crops:
        all_series[crop] = get_yearly_series(df, crop)

    print("TREND OF YEARLY MEDIAN YIELD (1997-2019)")
    print(trend_summary(all_series).to_string(index=False))

    errors = evaluate_methods(all_series)
    print("\nTEST ON 2016-2019 (trained on 1997-2015)")
    print(errors.to_string(index=False))

    all_forecasts = {}
    chosen = {}
    print("\nFORECAST FOR 2020-2024")
    for crop in crops:
        chosen[crop] = best_method(errors, crop)
        all_forecasts[crop] = future_forecast(all_series[crop], chosen[crop])
        print("\n" + crop + " (method: " + chosen[crop] + ")")
        print(all_forecasts[crop].to_string(index=False))

    plot_history(all_series)
    plot_holdout(all_series)
    plot_forecasts(all_series, all_forecasts, chosen)

    print_limitations()


if __name__ == "__main__":
    main()
