from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

folder = Path(__file__).resolve().parent
data_folder = folder / "dataset"
save_folder = folder / "results" / "statistical_analysis"
graph_folder = folder / "output_graphs" / "statistical_analysis"


def read_data(file, keys):
    data = pd.read_csv(data_folder / file)
    data.columns = data.columns.str.strip().str.lower()
    for name in data.select_dtypes(include="object"):
        data[name] = data[name].str.strip()
    if data.isnull().sum().sum() > 0 or data.duplicated(keys).sum() > 0:
        raise ValueError("Check missing or repeated records in " + file)
    if data.isin([float("inf"), float("-inf")]).sum().sum() > 0:
        raise ValueError("Check infinite numbers in " + file)
    return data


def get_stats(data):
    numbers = data.select_dtypes(include="number").drop(columns="year", errors="ignore")
    table = pd.DataFrame()
    table["count"] = numbers.count()
    table["mean"] = numbers.mean()
    table["median"] = numbers.median()
    table["mode"] = "No repeated value"
    table["mode_frequency"] = 1
    for name in numbers:
        count = numbers[name].value_counts().max()
        table.loc[name, "mode_frequency"] = count
        if count > 1:
            table.loc[name, "mode"] = "; ".join(numbers[name].mode().astype(str))
    table["variance"] = numbers.var()
    table["std_deviation"] = numbers.std()
    table["minimum"] = numbers.min()
    table["maximum"] = numbers.max()
    table.index.name = "variable"
    return table


def run_analysis():
    save_folder.mkdir(parents=True, exist_ok=True)
    crop = read_data("crop_yield.csv", ["crop", "year", "season", "state"])
    soil = read_data("state_soil_data.csv", ["state"])
    weather = read_data("state_weather_data_1997_2020.csv", ["state", "year"])

    tables = []
    for name, data in [("crop", crop), ("soil", soil), ("weather", weather)]:
        table = get_stats(data).reset_index()
        table.insert(0, "source", name)
        tables.append(table)
    pd.concat(tables).to_csv(save_folder / "descriptive_statistics.csv", index=False)
    crop.groupby("crop")["yield"].agg(
        ["count", "mean", "median", "var", "std", "min", "max"]
    ).to_csv(save_folder / "yield_by_crop.csv")
    crop.groupby("year").size().to_csv(save_folder / "year_coverage.csv", header=["records"])

    rice = crop[crop["crop"] == "Rice"]
    rice = rice[rice["season"] == "Kharif"]
    rice = rice[rice["year"] == 2019]
    rice = rice.merge(weather, on=["state", "year"], how="left", validate="one_to_one")
    rice = rice.merge(soil, on="state", how="left", validate="one_to_one")
    if rice.empty or rice.isnull().sum().sum() > 0:
        raise ValueError("Check the matching rice, soil and weather records.")
    rice = rice.sort_values("state")
    rice.to_csv(save_folder / "analysis_sample_2019.csv", index=False)
    get_stats(rice).to_csv(save_folder / "descriptive_statistics_2019.csv")

    numbers = rice.select_dtypes(include="number").drop(columns="year")
    pearson = numbers.corr()
    spearman = numbers.corr(method="spearman")
    pearson.to_csv(save_folder / "correlation_matrix_2019.csv")
    correlations = pd.DataFrame({
        "pearson_r": pearson["yield"], "spearman_rho": spearman["yield"], "n": len(rice)
    }).drop(index="yield")
    correlations.to_csv(save_folder / "yield_correlations_2019.csv")

    middle = rice["total_rainfall_mm"].median()
    rice["rainfall_group"] = "Higher"
    rice.loc[rice["total_rainfall_mm"] <= middle, "rainfall_group"] = "Lower_or_equal"
    low = rice[rice["rainfall_group"] == "Lower_or_equal"]["yield"]
    high = rice[rice["rainfall_group"] == "Higher"]["yield"]
    if len(low) < 2 or len(high) < 2:
        raise ValueError("Each rainfall group needs at least two records.")
    test = stats.ttest_ind(low, high, equal_var=False)
    interval = test.confidence_interval()
    if test.pvalue < 0.05:
        decision = "Reject H0"
        finding = "The groups show evidence of different mean yields under the test assumptions."
    else:
        decision = "Do not reject H0"
        finding = "There is not enough evidence of different mean yields."
    rice.groupby("rainfall_group")["yield"].agg(
        ["count", "mean", "median", "std", "min", "max", "skew"]
    ).to_csv(save_folder / "rainfall_group_summary.csv")
    rice[["state", "total_rainfall_mm", "yield", "rainfall_group"]].to_csv(
        save_folder / "rainfall_groups.csv", index=False
    )
    pd.DataFrame([{
        "test": "Welch independent two-sample t-test", "alternative": "two-sided",
        "rainfall_cutoff_mm": middle, "n_lower": len(low), "n_higher": len(high),
        "mean_lower": low.mean(), "mean_higher": high.mean(),
        "difference_lower_minus_higher": low.mean() - high.mean(),
        "t_statistic": test.statistic, "degrees_of_freedom": test.df,
        "p_value": test.pvalue, "alpha": 0.05, "decision": decision,
        "difference_ci95_lower": interval.low, "difference_ci95_upper": interval.high
    }]).to_csv(save_folder / "hypothesis_test.csv", index=False)

    graph_folder.mkdir(parents=True, exist_ok=True)
    names = [f"Lower/equal rainfall\n{len(low)} state/UT records", f"Higher rainfall\n{len(high)} state/UT records"]
    values = [low.mean(), high.mean()]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, values, color="#3B6FB6", width=0.55)
    plt.bar_label(bars, fmt="%.2f", padding=5)
    plt.title("Average rice yield by rainfall group\nKharif 2019")
    plt.ylabel("Average yield (dataset units)")
    plt.xlabel(f"Annual rainfall split at {middle:.2f} mm")
    plt.ylim(0, max(values) * 1.2)
    plt.figtext(0.5, 0.02, "Observed state averages; this does not prove rainfall causes yield.", ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(graph_folder / "rainfall_group_yield.png", dpi=160)
    plt.close()

    rates = crop[["year"]].copy()
    rates["fertilizer_per_area"] = (crop["fertilizer"] / crop["area"]).round(6)
    rates["pesticide_per_area"] = (crop["pesticide"] / crop["area"]).round(6)
    rates.groupby("year").nunique().to_csv(save_folder / "input_rate_checks.csv")

    report = f"""MEMBER 2 - STATISTICAL FINDINGS
Full dataset: {len(crop)} crop records. Original CSV files are unchanged.
The descriptive statistics files contain mean, median, mode, sample variance and standard deviation.
Each source is summarised separately. Different crop scales make the overall mean unsuitable as national yield.
All tied repeated modes are saved. No repeated value means no useful repeated mode.
Rice, Kharif, 2019: {len(rice)} state/UT records; average yield = {rice['yield'].mean():.6f}.
Correlations with yield:
{correlations.round(6).to_string()}
Positive means values tend to rise together; negative means they tend to move in opposite directions.
Near zero means little linear (Pearson) or rank (Spearman) relationship. These are not tests of causation.
H0: equal population mean yields in the two rainfall groups. H1: different population mean yields.
Rainfall split = {middle:.2f} mm, the sample median, not a recommended rainfall level.
Lower/equal group: {len(low)} records, mean = {low.mean():.6f}. Higher group: {len(high)} records, mean = {high.mean():.6f}.
Welch two-sided t-test: t = {test.statistic:.6f}, p = {test.pvalue:.8f}; cutoff = 0.05; {decision}.
95% interval for the mean difference: [{interval.low:.6f}, {interval.high:.6f}].
{finding}
The test assumes independent observations and approximately normal yields in the small groups.
States can share conditions; they are not a random sample, and other differences are uncontrolled.
2019 is used because 2020 has only 37 crop records. Findings concern this selected comparison.
Weather is annual; soil has no year. Haryana and Punjab have identical supplied weather series.
Fertilizer and pesticide per-area rates are shared within years, so their separate effects cannot be identified.
Yield values are used as supplied; units and data collection methods were not independently verified.
Do not conclude that reducing rainfall improves yield; irrigation and other possible explanations were not tested.
"""
    for file in ["hitans_findings.txt", "member2_findings.txt"]:
        (save_folder / file).write_text(report, encoding="utf-8")
    print(report)
    return rice, test, correlations


if __name__ == "__main__":
    run_analysis()
