# Validation

Source reviewed: `shatakshisankhe16112007/python_capstone`, main commit
`ee13ec8cb0220b75b60c4b91c906a1373d8a7f81`, checked 3 November 2026.

Read: `data_cleaning.py`, `visualization.py`, `common_utils.py`, `regression.py`,
`forecasting.py`, `main.py`, `statistical_analysis.py` and `requirements.txt`.
The current dataset folder still shows the same last change as the previously
verified download. Tests use those unchanged source CSVs.

- Ran the current GitHub Member 2 script as a reference.
- Ran the simplified replacement independently.
- Compared all 11 CSVs: same columns, row counts and numerical results.
  Numeric comparisons passed within 1e-12; repeated mode strings now preserve
  full precision instead of the original six-significant-digit formatting.
- Checked that the replacement has no comments, docstrings, lambdas,
  comprehensions or classes.
- Ran the unchanged current `main.py` with only `statistical_analysis.py`
  replaced in an isolated test folder. All five stages completed.
- Verified the other six Python files in that test folder remained unchanged.
- Verified earlier local Member 2 packages remained unchanged during testing.

Runtime: Python 3.12.14, NumPy 2.3.5, Pandas 2.2.3, SciPy 1.17.1,
Matplotlib 3.11.2, Seaborn 0.13.2 and Statsmodels 0.15.0.
Dependencies were installed in a separate test folder.

## Pipeline output

```text
CAPSTONE PROJECT EXECUTION SUMMARY
============================================================================
Step   Analytical Stage                       Author               Status     Runtime
----------------------------------------------------------------------------
Step 1  Data Cleaning & Exploration            Shatakshi Sankhe     SUCCESS    0.02s
Step 2  Data Visualizations (3 Dashboards)     Shatakshi Sankhe     SUCCESS    2.50s
Step 3  Statistical Analysis & Hypothesis Testing Hitans Sangrajka     SUCCESS    0.04s
Step 4  Multiple Linear Regression & Diagnostics Prathamesh Samel     SUCCESS    0.52s
Step 5  Time Series & ARIMA Forecasting        Prathamesh Samel     SUCCESS    1.37s
----------------------------------------------------------------------------
Total Pipeline Runtime: 4.46 seconds.
All output graphs saved to: output_graphs/
All numerical summaries saved to: results/
============================================================================
```

Successful execution verifies integration. It is not an independent validation
of every modeling choice or interpretation in the other members' modules.
