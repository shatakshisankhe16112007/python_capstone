"""Statistical Analysis, correlations, and hypothesis testing.
Author: Hitans Sangrajka (Python Programming Laboratory)

Run: python statistical_analysis.py
Inputs stay unchanged. Results are saved under results/statistical_analysis/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results" / "statistical_analysis"


def load_csv(filename, keys):
    """Read a copy, standardise names and check the expected row identifiers."""
    data = pd.read_csv(DATA_DIR / filename)
    data.columns = data.columns.str.strip().str.lower()
    for column in data.select_dtypes(include=["object", "string"]):
        data[column] = data[column].str.strip()
    if data.isna().any().any() or data.duplicated(keys).any():
        raise ValueError(f"Check missing values or duplicate keys in {filename}.")
    numeric = data.select_dtypes(include=np.number)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"Check infinite numeric values in {filename}.")
    return data


def descriptive_statistics(data):
    """Use sample variance/SD (n - 1); show every tied repeated mode."""
    rows = []
    for column in data.select_dtypes(include=np.number):
        if column == "year":  # A time label, not an agricultural measurement.
            continue
        values = data[column]
        frequency = int(values.value_counts().max())
        mode = "; ".join(f"{v:g}" for v in values.mode())
        if frequency == 1:
            mode = "No repeated value"
        rows.append({
            "variable": column, "count": len(values),
            "mean": values.mean(), "median": values.median(),
            "mode": mode, "mode_frequency": frequency,
            "variance": values.var(ddof=1), "std_deviation": values.std(ddof=1),
            "minimum": values.min(), "maximum": values.max(),
        })
    return pd.DataFrame(rows).set_index("variable")


def run_analysis():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    crop = load_csv("crop_yield.csv", ["crop", "year", "season", "state"])
    soil = load_csv("state_soil_data.csv", ["state"])
    weather = load_csv("state_weather_data_1997_2020.csv", ["state", "year"])

    # 1. Summarise each source at its own level. Do not repeat soil/weather
    # rows across crops before describing what is in the provided files.
    summaries = pd.concat({
        "crop": descriptive_statistics(crop),
        "soil": descriptive_statistics(soil),
        "weather": descriptive_statistics(weather),
    }, names=["source", "variable"])
    summaries.to_csv(RESULTS_DIR / "descriptive_statistics.csv")
    crop.groupby("crop")["yield"].agg(
        ["count", "mean", "median", "var", "std", "min", "max"]
    ).to_csv(RESULTS_DIR / "yield_by_crop.csv")
    crop.groupby("year").size().rename("records").to_csv(
        RESULTS_DIR / "year_coverage.csv"
    )

    # 2. A comparable subset: one crop, season and year, one row per state.
    # 2019 is used because 2020 has only 37 records in the supplied crop file.
    focus = crop[
        (crop["crop"] == "Rice") & (crop["season"] == "Kharif")
        & (crop["year"] == 2019)
    ].copy()
    focus = focus.merge(
        weather, on=["state", "year"], how="left", validate="one_to_one"
    ).merge(soil, on="state", how="left", validate="one_to_one")
    if focus.empty or focus.isna().any().any():
        raise ValueError("The Rice-Kharif 2019 sample is empty or has unmatched data.")
    focus = focus.sort_values("state")
    focus.to_csv(RESULTS_DIR / "analysis_sample_2019.csv", index=False)
    focus_stats = descriptive_statistics(focus)
    focus_stats.to_csv(RESULTS_DIR / "descriptive_statistics_2019.csv")

    # Pearson measures linear association; Spearman compares ranks.
    numeric = focus.select_dtypes(include=np.number).drop(columns="year")
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    pearson.to_csv(RESULTS_DIR / "correlation_matrix_2019.csv")
    correlations = pd.DataFrame({
        "pearson_r": pearson["yield"], "spearman_rho": spearman["yield"]
    }).drop(index="yield")
    correlations["n"] = len(focus)
    correlations.to_csv(RESULTS_DIR / "yield_correlations_2019.csv")

    # 3. One two-sided Welch t-test, with alpha fixed at 0.05.
    # Split at the sample rainfall median, not at a threshold chosen for a p-value.
    alpha = 0.05
    cutoff = focus["total_rainfall_mm"].median()
    focus["rainfall_group"] = np.where(
        focus["total_rainfall_mm"] <= cutoff, "Lower_or_equal", "Higher"
    )
    lower = focus.loc[focus["rainfall_group"] == "Lower_or_equal", "yield"]
    higher = focus.loc[focus["rainfall_group"] == "Higher", "yield"]
    if min(len(lower), len(higher)) < 2:
        raise ValueError("Each rainfall group needs at least two observations.")
    test = stats.ttest_ind(lower, higher, equal_var=False, alternative="two-sided")
    interval = test.confidence_interval(confidence_level=0.95)
    decision = "Reject H0" if test.pvalue < alpha else "Do not reject H0"
    groups = focus.groupby("rainfall_group")["yield"].agg(
        ["count", "mean", "median", "std", "min", "max", "skew"]
    )
    groups.to_csv(RESULTS_DIR / "rainfall_group_summary.csv")
    focus[["state", "total_rainfall_mm", "yield", "rainfall_group"]].to_csv(
        RESULTS_DIR / "rainfall_groups.csv", index=False
    )
    pd.DataFrame([{
        "test": "Welch independent two-sample t-test", "alternative": "two-sided",
        "rainfall_cutoff_mm": cutoff, "n_lower": len(lower), "n_higher": len(higher),
        "mean_lower": lower.mean(), "mean_higher": higher.mean(),
        "difference_lower_minus_higher": lower.mean() - higher.mean(),
        "t_statistic": test.statistic, "degrees_of_freedom": test.df,
        "p_value": test.pvalue, "alpha": alpha, "decision": decision,
        "difference_ci95_lower": interval.low, "difference_ci95_upper": interval.high,
    }]).to_csv(RESULTS_DIR / "hypothesis_test.csv", index=False)

    # 4. Small data checks that explain the limits of the results.
    rates = crop.assign(
        fertilizer_per_area=crop["fertilizer"] / crop["area"],
        pesticide_per_area=crop["pesticide"] / crop["area"],
    )
    rate_counts = rates.groupby("year")[
        ["fertilizer_per_area", "pesticide_per_area"]
    ].agg(lambda values: values.round(6).nunique())
    rate_counts.to_csv(RESULTS_DIR / "input_rate_checks.csv")
    pooled = summaries.loc[("crop", "yield")]
    selected = focus_stats.loc["yield"]
    result_sentence = (
        "The group mean yields differ at the 5% level under the test assumptions."
        if test.pvalue < alpha else
        "There is insufficient evidence of different group mean yields at the 5% level."
    )
    report = f"""HITANS SANGRAJKA - STATISTICAL FINDINGS

Scope
Crop file: {len(crop):,} rows; {crop['crop'].nunique()} crops;
{crop['state'].nunique()} state/UT labels; {crop['year'].min()}-{crop['year'].max()}.
Soil file: {len(soil)} state rows. Weather file: {len(weather)} state-year rows.
No missing cells or duplicate analysis keys were found after trimming text.
All {len(focus)} selected crop rows matched both the soil and weather files.
Original CSV files were not modified; no outliers were removed or values imputed.

1. DESCRIPTIVE STATISTICS
All crop records, supplied yield values:
Mean = {pooled['mean']:.6f}; median = {pooled['median']:.6f};
mode = {pooled['mode']}; sample variance = {pooled['variance']:.6f};
sample standard deviation = {pooled['std_deviation']:.6f}.
The large mean-median gap and SD show that high values dominate this pooled
summary. Crop scales differ, so this is NOT a national agricultural yield.
See yield_by_crop.csv for separate crop summaries.

Rice, Kharif, 2019 ({len(focus)} states, equal weight per state):
Mean = {selected['mean']:.6f}; median = {selected['median']:.6f};
mode = {selected['mode']}; sample variance = {selected['variance']:.6f};
sample standard deviation = {selected['std_deviation']:.6f}.
These summaries retain the dataset's yield units. Variance has squared units.
No repeated value means no useful repeated mode; the code does not select
an arbitrary first value. All tied repeated modes are retained.

2. CORRELATIONS WITH YIELD - RICE, KHARIF, 2019
{correlations.to_string(float_format=lambda value: f'{value:.6f}')}

Positive coefficients mean higher values tend to occur together; negative
coefficients mean higher values tend to accompany lower yield. Values near
zero indicate little linear (Pearson) or monotonic (Spearman) association.
Rainfall has a negative association; N and P have positive associations.
K has little linear association. These are descriptive, unadjusted comparisons:
no significance claim is made for the individual correlation coefficients.
Fertilizer and pesticide totals are area-scaled yearly rates in this dataset.
Their correlations therefore cannot separate application effects from area.
Production and yield are also related agricultural quantities, not independent
experimental treatments. None of these coefficients proves causation.

3. HYPOTHESIS TEST
Question: Does mean Rice-Kharif 2019 yield differ between the two rainfall groups?
H0: The two groups have equal population mean yield.
H1: The two groups have different population mean yields.
Test: two-sided Welch independent two-sample t-test; alpha = {alpha}.
Lower/equal rainfall: <= {cutoff:.2f} mm; higher rainfall: > {cutoff:.2f} mm.
This is the selected sample's median, not an agronomic rainfall threshold.
Group sizes: {len(lower)} and {len(higher)}.
Group mean yields: {lower.mean():.6f} and {higher.mean():.6f}.
Difference (lower minus higher): {lower.mean() - higher.mean():.6f}.
95% confidence interval for the mean difference: [{interval.low:.6f}, {interval.high:.6f}].
t = {test.statistic:.6f}; df = {test.df:.6f}; p = {test.pvalue:.8f}.
Decision: {decision}. {result_sentence}
For these data the lower-rainfall group has the larger observed mean.
This does NOT show that reducing rainfall would improve yield.

Why this test and subset?
One crop, season and year makes the comparison more coherent than mixing crops.
Each state appears once; repeated years are not treated as independent samples.
Welch compares two means without assuming equal group variances.
The small groups require approximately normal yield distributions without
extreme outliers; the saved group summaries show only mild sample skewness.
Normality is not established by that check. State independence is an assumption:
nearby states and shared policies/weather can produce dependence.

4. LIMITATIONS
This is an exploratory observational classroom analysis, not a causal estimate.
The available states are not a random sample; generalisation is limited.
Only 22 states are represented in the chosen subset, not all 30 source labels.
2020 is sparse (37 crop rows), so 2019 is used for the focused analysis.
Soil has one fixed value per state with no observation year. Weather is annual,
not Kharif-specific. Measurement provenance and all crop/soil units are unverified.
Haryana and Punjab have identical weather series in the supplied weather file;
their values are retained, but this further limits source-quality confidence.
Supplied yield often differs from production/area, so it is not recalculated.
Irrigation, crop varieties and other state differences are not controlled.
The median split loses rainfall detail; continuous correlations are also shown.
Statistical significance does not establish practical importance or causation.

Method reference:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
"""
    (RESULTS_DIR / "hitans_findings.txt").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "member2_findings.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved results to: {RESULTS_DIR}")
    return focus, test, correlations


if __name__ == "__main__":
    run_analysis()
