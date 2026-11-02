# Member 2 — Statistical Analysis

**Member 2 statistical analysis, generated results and run instructions.**

This package completes the four Member 2 tasks from the shared assignment:

| Task | Where it is completed |
|---|---|
| Mean, median, mode, variance and standard deviation | `descriptive_statistics.csv` and `descriptive_statistics_2019.csv` |
| Correlation analysis | `yield_correlations_2019.csv` and `correlation_matrix_2019.csv` |
| A suitable hypothesis test | `hypothesis_test.csv` and the rainfall-group tables |
| Interpretation of the statistical results | `results/member2/member2_findings.txt` |

The Python code is in **`statistical_analysis.py`**. It uses ordinary Pandas
operations and one SciPy t-test. The latter half of the file writes the explanation.
It does not require graphs, machine learning, regression or forecasting.

## Start here

To read the completed work, open [the statistical findings](results/member2/member2_findings.txt).
For a beginner-friendly walkthrough and viva questions, read
[the Member 2 viva guide](MEMBER2_VIVA_GUIDE.md).
All results have already been generated from the supplied CSVs.

To run it again, open a terminal in the folder containing `statistical_analysis.py`
(the project root in the GitHub repository):

```bash
python -m pip install -r requirements_member2.txt
python statistical_analysis.py
```

On a Mac where the command is `python3`, use `python3` in both commands.
Python 3.9 or later is recommended. Tested with Python 3.12.14, Pandas 2.2.3,
NumPy 2.3.5 and SciPy 1.17.1.

You can also open `statistical_analysis.py` in VS Code or another Python editor
and run it. Keep the `dataset` folder beside it. Paths are based on the script's
location, so no personal Windows/Mac path needs to be edited.

## Files

```text
python_capstone/  (Member 2 files shown)
├── statistical_analysis.py
├── requirements_member2.txt
├── README_MEMBER2.md
├── MEMBER2_VIVA_GUIDE.md
├── dataset/
│   ├── crop_yield.csv
│   ├── state_soil_data.csv
│   └── state_weather_data_1997_2020.csv
└── results/member2/
    ├── member2_findings.txt
    ├── descriptive_statistics.csv
    ├── yield_by_crop.csv
    ├── year_coverage.csv
    ├── analysis_sample_2019.csv
    ├── descriptive_statistics_2019.csv
    ├── correlation_matrix_2019.csv
    ├── yield_correlations_2019.csv
    ├── rainfall_groups.csv
    ├── rainfall_group_summary.csv
    ├── hypothesis_test.csv
    └── input_rate_checks.csv
```

The CSVs in `dataset/` are unchanged copies of the three supplied project files.
The script trims text and standardises column names only in memory.
No records or outliers are deleted. Rerunning replaces only the generated
Member 2 results.

## Main results

For all 19,689 supplied crop records, the yield mean is **79.954009**, median
**1.030000**, mode **1**, sample variance **771,421.768059**, and sample standard
deviation **878.306193**. These are unweighted summaries of the supplied numbers.
Different crops have very different scales; this mean is **not national crop
productivity**. Separate summaries for all 55 crops are also provided.

For a more comparable analysis, correlations and the hypothesis test use
**Rice, Kharif, 2019**: 22 state/UT observations, with one observation per label.
Rice is the most frequent crop in the source, Kharif is its most frequent season,
and 2019 is used because 2020 has only 37 crop records. Eight of the 30 source
state/UT labels are not represented in this particular subset.

| Yield statistic for the selected 22 observations | Value |
|---|---:|
| Mean | 2.417356 |
| Median | 2.330795 |
| Mode | No repeated value |
| Sample variance | 0.589812 |
| Sample standard deviation | 0.767992 |

Pearson correlations with yield include **rainfall −0.570973**, **soil N
+0.620151**, **soil P +0.624994** and **soil K +0.006408**. Spearman correlations
are included as a simple comparison using ranks. These are exploratory
associations, without significance tests for each individual coefficient.

**Hypothesis test:** Compare mean yield in states with rainfall at or below
the selected sample's median of **1,312.03 mm**, versus states above it.
This gives 11 observations per group.

- **H₀:** The two groups have equal population mean yield.
- **H₁:** The two groups have different population mean yields.
- **Method:** Two-sided Welch independent two-sample t-test, α = 0.05.
- **Mean yields:** 2.871385 for lower/equal rainfall and 1.963327 for higher rainfall.
- **Result:** t = 3.398987; df = 17.474376; p = **0.00330847**.
- **Decision:** Reject H₀ under the test assumptions. The groups differ in mean
  yield in this exploratory analysis.
- **Mean difference:** 0.908058, with a 95% confidence interval of
  [0.345573, 1.470543], in the supplied yield units.

Welch's test is used because the two groups need not have equal variances.
It still assumes independent observations and approximately normal group
distributions for these small samples. The group summaries show mild sample
skewness, which is a limited check rather than proof of normality. States can
share regional influences, so independence and the resulting p-value are
approximate. The sample median is a convenient grouping rule, not a recommended
rainfall level. The result does **not** mean that reducing rainfall improves yield.

## Simple explanation for your presentation or viva

> My part was statistical analysis. I calculated mean, median, mode, sample
> variance and standard deviation, then examined correlations between yield
> and the agricultural variables. Since different crops and years are not
> directly comparable, I used Rice in Kharif 2019 for the focused analysis.
> I compared two rainfall groups using Welch's t-test. The p-value was about
> 0.0033, below 0.05, so I rejected the equal-means null hypothesis under the
> test assumptions. However, this is an observational association. Irrigation,
> geography and other factors could explain the difference.

Useful definitions:

- **Mean:** Arithmetic average.
- **Median:** Middle value after sorting.
- **Mode:** Most frequent repeated value. If every value occurs once, this
  project reports “No repeated value.” All tied repeated modes are retained.
- **Variance:** Squared spread around the mean. Sample variance divides by n − 1.
- **Standard deviation:** Square root of variance, in the original units.
- **Pearson correlation:** Strength and direction of a linear relationship.
- **Spearman correlation:** Strength and direction of a relationship using ranks.
- **p-value:** Under H₀ and the test assumptions, the probability of a test
  statistic at least as extreme as the observed statistic; it is not the
  probability that H₀ is true.

## Data choices and limitations to remember

1. Soil summaries use the 30 original soil rows; weather summaries use the 720
   original state-year rows. These measurements are not repeated across crop
   records before calculating their descriptive statistics.
2. Weather is joined by `state` and `year`; soil is joined by `state`. The
   selected 22 records match both files, with no row multiplication.
3. Fertilizer and pesticide totals equal area multiplied by a common annual
   rate, to the supplied precision. In 2019 those rates are 171.76 and 0.37.
   Their correlations therefore cannot identify independent application effects.
4. Soil has no measurement year, and weather is annual rather than
   season-specific. Haryana and Punjab have identical supplied weather series.
   Data collection methods and crop/soil units were not independently verified.
5. The supplied `yield` column often differs from `production / area`. It is
   preserved as provided; no units or alternative definition are invented.
6. These are state summaries, not farm-level experimental observations.
   The analysis neither identifies causal factors nor establishes conclusions
   for every Indian state, crop or year.

## Integration with the team project

`statistical_analysis.py` runs independently from the project root and reads the
three files in `dataset/`. It writes only to `results/member2/`.
`requirements_member2.txt` lists its dependencies separately from any dependencies
used by the other members.

The team's main entry point can call the analysis as follows:

```python
from statistical_analysis import run_analysis

if __name__ == "__main__":
    run_analysis()
```

The standalone Member 2 analysis has been run and its calculations checked.
Integration into a complete team pipeline has not been tested.

## Sources

- [Team repository](https://github.com/shatakshisankhe16112007/python_capstone)
  and the provided Member 2 task allocation.
- [Dataset page cited by the project](https://www.kaggle.com/datasets/anshumish/crop-yield-data-with-soil-and-weather-dataset).
  The actual analysis uses the three supplied project CSVs.
- [SciPy documentation for `ttest_ind`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html):
  `equal_var=False` selects Welch's t-test.
