# Member 2: statistical analysis

Member 2 uses `statistical_analysis.py` to summarise the data, check relationships with crop yield and compare two rainfall groups.

## How to run

From the project folder:

```bash
python -m pip install -r requirements.txt
python statistical_analysis.py
```

Use `python main.py` to run the complete team project. Results are saved in `results/statistical_analysis/`.

## Simple code flow

1. `read_data()` opens each CSV, trims spaces and checks missing or repeated records.
2. `get_stats()` calculates the mean, median, mode, sample variance and standard deviation, plus counts and ranges.
3. `run_analysis()` saves separate summaries for crop, soil and weather data.
4. `rice` selects Rice, Kharif and 2019, then joins matching soil and weather records.
5. `pearson` and `spearman` measure how the factors move with yield.
6. `low` and `high` contain yields from the two rainfall groups. A Welch t-test compares their means.

There are no handwritten statistical formulas or code comments. Pandas and SciPy do the calculations. The code uses ordinary functions, loops, lists and `if` statements. Direct arithmetic is used only for input rates and the difference between group means.

## Results to explain

The focused comparison contains 22 state/UT records, with mean yield 2.417356. Rainfall has a negative Pearson correlation with yield (-0.570973); soil nitrogen and phosphorus have positive correlations.

Splitting rainfall at the sample median, 1312.03 mm, gives 11 records per group. Average yields are 2.871385 for the lower/equal rainfall group and 1.963327 for the higher group. The two-sided Welch test gives p = 0.00330847, below the 0.05 cutoff. Under its assumptions, this supports a difference between the group means.

This does not prove rainfall caused the difference or that less water improves yield. Weather is annual, soil has no year, the states are not a random sample, and irrigation was not tested. Yield units are used as supplied and were not independently verified.

## How this fits the team

Member 1 explores and visualises the full crop dataset. Member 2 summarises all three sources and uses the 2019 rice comparison for correlation and hypothesis testing. Member 3 uses multiple historical years for regression and forecasting. The members can use different subsets for these different questions.

The simplified script keeps `run_analysis()`, its three return values, all 11 CSV filenames and columns, and both findings reports. Numerical results are preserved; mode number formatting and report wording are clearer. All five stages passed in an isolated run of the current team project. See [validation](results/statistical_analysis/VALIDATION_MEMBER2.md).
