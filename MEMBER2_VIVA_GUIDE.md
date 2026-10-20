# Member 2: Statistical Analysis Viva Guide

This guide explains [statistical_analysis.py](statistical_analysis.py):
descriptive statistics, correlations, a hypothesis test, and interpretations.
The detailed report is
[member2_findings.txt](results/member2/member2_findings.txt).

## 1. Libraries and code flow

| Library | Role in this program |
|---|---|
| `pathlib` | Locates input and output folders relative to the Python file. |
| `pandas` | Reads CSVs, filters rows, joins tables, calculates statistics, and saves results. |
| `numpy` | Checks that numbers are finite and assigns rainfall-group labels. |
| `scipy.stats` | Runs Welch's t-test and calculates its confidence interval. |

1. **Read three CSVs.** Crop data has 19,689 records, soil has 30 state rows,
   and weather has 720 state-year rows.
2. **Check the inputs.** `load_csv()` makes column names lowercase, trims
   surrounding spaces, and checks missing cells, duplicate identifying keys,
   and infinite numbers. It raises an error if a check fails.
3. **Summarise each source separately.** Soil and weather statistics use their
   original rows, avoiding repeated counting. Separate crop summaries are saved.
4. **Select Rice, Kharif, 2019.** Join weather and soil information to these
   records, then calculate focused descriptive statistics and correlations.
5. **Compare two rainfall groups.** Run one two-sided Welch t-test.
6. **Save tables and interpretations** under `results/member2/`.

Original CSVs remain unchanged. Preparation happens in memory; no outliers are
removed or missing measurements filled.

## 2. Understanding the descriptive measures

Consider the values **[2, 3, 3, 4, 8]**:

| Measure | Meaning | Example |
|---|---|---|
| Mean | Sum divided by number of values | 20 ÷ 5 = **4** |
| Median | Middle value after sorting | **3** |
| Mode | Most frequent value | **3**, occurring twice |
| Sample variance | Sum of squared deviations from the mean, divided by n−1 | 22 ÷ 4 = **5.5** |
| Sample standard deviation | Square root of sample variance | √5.5 ≈ **2.345** |

Variance measures spread in squared units; standard deviation uses the original
units. `ddof=1` specifies the
**n−1** denominator for sample variance and standard deviation.

All tied repeated modes are kept. If every value appears once, the report says
**“No repeated value”**. The `year` column is excluded because it identifies time.

## 3. Overall and focused yield results

| Yield statistic | All crop records | Rice–Kharif 2019 |
|---|---:|---:|
| Observations | 19,689 | 22 |
| Mean | 79.954009 | 2.417356 |
| Median | 1.030000 | 2.330795 |
| Mode | 1 | No repeated value |
| Sample variance | 771,421.768059 | 0.589812 |
| Sample standard deviation | 878.306193 | 0.767992 |

Some very large values raise the overall mean far above the median. Crops have
different scales, so this is **not a national agricultural yield**. The focused
table compares one crop, season, and year, weighting included states/UTs equally.

See [overall statistics](results/member2/descriptive_statistics.csv) and
[focused statistics](results/member2/descriptive_statistics_2019.csv).

## 4. Why this subset and these joins?

Rice is the most frequent crop in the source, and Kharif is its most frequent
season. The analysis uses 2019 because the entire 2020 crop data contains only
37 records. The selected sample has **22 state/UT labels**, with one row per
label; eight source labels are absent from this subset.

Weather is joined using **state and year**, because weather changes over time.
Soil is joined using **state**, because its file has no year column.
`validate="one_to_one"` prevents row multiplication. All 22 rows match both files.

This improves comparability and avoids treating repeated state-years as
independent observations. Other state differences remain uncontrolled.

## 5. Interpreting correlations

**Pearson correlation** measures linear association. **Spearman correlation**
uses ranks to measure monotonic association: whether higher values generally
go with higher or lower values of another variable.

Both range from **−1 to +1**. Positive means the variables tend to increase
together; negative means higher values tend to accompany lower yield. Near
zero indicates little association of the particular type measured.

| Variable versus yield | Pearson correlation |
|---|---:|
| Annual rainfall | −0.570973 |
| Soil N | +0.620151 |
| Soil P | +0.624994 |
| Soil K | +0.006408 |

Rainfall has a negative association with yield in this sample, while N and P
have positive associations. K has little linear association. These coefficients
are descriptive; the program does not test each one's statistical significance.

**−0.571 does not mean a 57.1% yield reduction.** Correlation is not a percentage
effect, and it does not prove causation. See
[yield correlations](results/member2/yield_correlations_2019.csv).

## 6. The hypothesis test

**Question:** Does mean Rice–Kharif 2019 yield differ between two rainfall groups?

The sample median rainfall is **1,312.03 mm**:

- Lower/equal group: rainfall ≤ 1,312.03 mm; **11 observations**.
- Higher group: rainfall > 1,312.03 mm; **11 observations**.

This cutoff is not an agricultural recommendation.

**H₀:** The groups have equal population mean yield.  
**H₁:** The groups have different population mean yields.

The code uses `stats.ttest_ind(lower, higher, equal_var=False,
alternative="two-sided")`. `equal_var=False` selects **Welch's test**, which
does not assume equal group variances. “Two-sided” means a difference in either
direction is considered.

| Result | Value |
|---|---:|
| Lower/equal rainfall mean yield | 2.871385 |
| Higher rainfall mean yield | 1.963327 |
| Difference: lower minus higher | 0.908058 |
| t-statistic | 3.398987 |
| Degrees of freedom | 17.474376 |
| p-value | 0.00330847 |
| Significance level, α | 0.05 |
| 95% confidence interval for the difference | [0.345573, 1.470543] |

Since **p < 0.05**, reject H₀ under the test assumptions. The groups show evidence
of different mean yields. The lower-rainfall group has the higher observed mean.

The p-value measures how unusual a result this extreme or more extreme would
be **if H₀ and the assumptions held**. It is not the probability that H₀ is true.
The interval describes uncertainty in the mean difference, not a range
containing 95% of individual yields.

## 7. Limitations to explain

- Groups are small. Approximately normal yield distributions without extreme
  outliers are needed for this t-test approximation. Mild observed skewness
  does not prove normality.
- States can share weather and policies, so independence is an assumption.
  The covered states are not a random sample.
- Soil values are fixed per state; weather is annual, not Kharif-specific.
  Haryana and Punjab have identical supplied weather series.
- Fertilizer and pesticide totals are area multiplied by common yearly rates.
  Their correlations cannot isolate application effects.
- Yield units and collection methods are unverified. Supplied yield often
  differs from production ÷ area, so it is retained unchanged.
- Irrigation and other differences are uncontrolled. Statistical significance
  does not establish causation or practical importance.

## 8. Python syntax to recognise

| Code | Meaning |
|---|---|
| `def run_analysis():` | Defines the main analysis function. |
| `data[condition]` | Selects rows satisfying a condition. |
| `&` between conditions | Requires both conditions to be true. |
| `.copy()` | Creates a separate working DataFrame. |
| `.groupby(...).agg(...)` | Calculates summaries for each group. |
| `.to_csv(...)` | Saves a table as a CSV. |
| `if __name__ == "__main__":` | Runs the analysis when the file is executed directly. |

## 9. Short viva speech and questions

> My contribution is statistical analysis. I calculated descriptive measures,
> examined correlations, and compared rainfall groups using Welch's t-test.
> I used Rice in Kharif 2019 to improve comparability. The p-value was about
> 0.0033, so I rejected equal group means at the 5% level under the assumptions.
> However, these observational results do not establish a causal rainfall effect.

**Why Welch?** It compares two means without requiring equal variances.  
**Why n−1?** The code reports sample variance and standard deviation.  
**Can we recommend less rainfall?** No; other state differences could explain
the association.  
**Were source values changed?** No; original CSVs and supplied yield values
were preserved.  
**Where are the results?** In `results/member2/`, including the
[hypothesis-test table](results/member2/hypothesis_test.csv) and findings report.
