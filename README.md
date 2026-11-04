# Analysis of Agricultural Yield Using Soil and Weather Parameters

**Course:** Python Programming Laboratory  
**Institution:** K. J. Somaiya School of Engineering, Somaiya Vidyavihar University, Mumbai  
**Academic Year:** 2026–27  
**Group:** Group 5  
**GitHub Repository:** [https://github.com/shatakshisankhe16112007/python_capstone](https://github.com/shatakshisankhe16112007/python_capstone)

---

## 1. Project Introduction

Agriculture is the primary livelihood source for a majority of India's population. Crop yield productivity is governed by complex agronomic interactions among crop genetic varieties, cultivation area, chemical inputs (fertilizers and pesticides), baseline soil chemistry (Nitrogen, Phosphorus, Potassium, and pH), and annual weather patterns (temperature, precipitation, and relative humidity).

This Capstone Project develops an end-to-end, reproducible computational pipeline in Python that ingests, cleans, explores, visualizes, statistically analyzes, models, and forecasts agricultural crop yields across 30 Indian states and Union Territories over a 24-year historical span (1997–2020). The implementation strictly applies the foundational concepts, procedures, and libraries taught across all 12 laboratory practicals of our Python Programming Laboratory syllabus.

---

## 2. Team Members & Contributions

| Student Name | Module / Role | Primary Responsibilities | Practical Alignment |
|---|---|---|---|
| **Shatakshi Sankhe** | `data_cleaning.py`<br>`visualization.py` | Dataset loading, structural exploration, regional mapping, and 3 comprehensive four-panel Matplotlib & Seaborn dashboards. | Exp 8 (Pandas), Exp 9 (Matplotlib), Exp 10 (Seaborn) |
| **Hitans Sangrajka** | `statistical_analysis.py` | Descriptive statistics across all sources, Rice-Kharif 2019 sample selection, Pearson & Spearman correlations, and Welch's independent two-sample $t$-test. | Exp 5 (Statistics), Exp 8 (Pandas), Exp 11 (SciPy stats) |
| **Prathamesh Samel** | `common_utils.py`<br>`regression.py`<br>`forecasting.py`<br>`main.py` | Shared utilities, Multiple Linear Regression via SciPy Normal Equations (`scipy.linalg.solve`), time-series stationarity (ADF), differencing, SMA/EWMA, Statsmodels ARIMA, 5-year forecasts, and master pipeline coordination. | Exp 6 (Modules), Exp 7 (NumPy), Exp 11 (SciPy linalg), Exp 12 (Statsmodels Time Series) |

---

## 3. Project Directory Structure

```text
python_capstone/
├── dataset/
│   ├── crop_yield.csv                      # Main agricultural dataset (19,689 rows, 9 cols)
│   ├── state_soil_data.csv                 # State soil chemistry dataset (30 rows, 5 cols)
│   └── state_weather_data_1997_2020.csv    # Annual state weather dataset (720 rows, 5 cols)
├── output_graphs/
│   ├── visualization/                      # 3 four-panel dashboards (12 subplots) by Shatakshi Sankhe
│   ├── statistical_analysis/               # Rainfall-group yield comparison bar chart by Hitans Sangrajka
│   ├── regression/                         # 3 regression scatter, correlation & residual plots
│   └── forecasting/                        # 4 time series stationarity, smoothing & ARIMA forecast plots
├── results/
│   ├── statistical_analysis/               # Descriptive stats, correlation matrices & Welch t-test results
│   ├── regression/                         # Model evaluation table & findings summary
│   └── forecasting/                        # ADF tests, smoothing, holdout & future forecast CSVs
├── common_utils.py                         # Reusable loaders, plot helpers & Normal Equations solver
├── data_cleaning.py                        # Dataset loading and structural exploration
├── visualization.py                        # Regional dashboards and agricultural factor plots
├── statistical_analysis.py                 # Descriptive statistics & Welch's independent t-test
├── regression.py                           # First-principles Normal Equations regression & diagnostics
├── forecasting.py                          # Statsmodels ARIMA time series forecasting & smoothing
├── main.py                                 # Master pipeline execution entry point
├── requirements.txt                        # Standardized dependencies (NumPy, Pandas, etc.)
└── README.md                               # Project documentation and user guide
```

---

## 4. Dataset Descriptions

The project utilizes three datasets located in the `dataset/` directory:

1. **`crop_yield.csv` (19,689 records × 9 columns)**
   - *Columns:* `crop`, `year`, `season`, `state`, `area`, `production`, `fertilizer`, `pesticide`, `yield`.
   - *Coverage:* Spans 1997 to 2020 across 30 Indian states and union territories for 55 distinct crop types.
   - *Source:* [Kaggle: Crop Yield Data with Soil and Weather Dataset](https://www.kaggle.com/datasets/anshumish/crop-yield-data-with-soil-and-weather-dataset).
2. **`state_soil_data.csv` (30 records × 5 columns)**
   - *Columns:* `state`, `N` (Nitrogen), `P` (Phosphorus), `K` (Potassium), `pH`.
   - *Coverage:* Baseline macronutrient and soil pH values for each of the 30 Indian states.
3. **`state_weather_data_1997_2020.csv` (720 records × 5 columns)**
   - *Columns:* `state`, `year`, `avg_temp_c`, `total_rainfall_mm`, `avg_humidity_percent`.
   - *Coverage:* Exactly 30 states × 24 years (1997–2020) = 720 rows.

---

## 5. Technology Stack & Installation

The project strictly utilizes the 6 permitted standard analytical libraries without external machine learning frameworks:
- `numpy >= 1.24.0` (Array operations and matrix algebra)
- `pandas >= 2.0.0` (DataFrame manipulation, indexing, and joins)
- `matplotlib >= 3.7.0` (2D plotting, subplots, and layouts)
- `seaborn >= 0.12.0` (Statistical distributions and regression plots)
- `scipy >= 1.11.0` (`scipy.linalg.solve` Normal Equations and statistical tests)
- `statsmodels >= 0.14.0` (ADF test and ARIMA time series modeling)

Install all dependencies using:
```bash
python -m pip install -r requirements.txt
```

---

## 6. How to Run the Project

### Running the Complete Pipeline
To execute all five stages sequentially:
```bash
python main.py
```

### Running Modules Independently
Each module can also be run independently:
```bash
python data_cleaning.py
python visualization.py
python statistical_analysis.py
python regression.py
python forecasting.py
```

---

## 7. Analytical Modules & Methodology

### Module 1: `data_cleaning.py` (Shatakshi Sankhe)
- Loads the primary crop dataset and performs programmatic exploration: `.head()`, `.shape`, `.columns`, `.dtypes`, `.isnull().sum()`, `.duplicated().sum()`, and `.describe()`.
- Verifies structural integrity across all records.

### Module 2: `visualization.py` (Shatakshi Sankhe)
- Constructs 3 comprehensive four-panel dashboards (12 subplots total) based on a custom regional grouping (North, South, East, West, Central, Northeast):
  1. `figure1_regional_production.png`: Regional Crop Production Analysis (Total production by region, Top 6 crops across regions, Average crop production heatmap, and Seasonal production by region).
  2. `figure2_agricultural_relationships.png`: Agricultural Factor Relationships (Fertilizer vs. yield, Pesticide vs. yield, Cultivated area vs. production bubble chart, and Correlation heatmap).
  3. `figure3_yield_seasonal_analysis.png`: Crop Yield and Seasonal Analysis (Yield distribution across regions, Yield across seasons violin plot, Yearly yield trends across regions, and Regional yield by season boxplots).
- The 3 standardized four-panel dashboards contributed by Shatakshi (`figure1_regional_production.png`, `figure2_agricultural_relationships.png`, and `figure3_yield_seasonal_analysis.png`) are saved in `output_graphs/` and mirrored in `output_graphs/visualization/`.

### Module 3: `statistical_analysis.py` (Hitans Sangrajka)
- Computes parametric and non-parametric descriptive statistics (mean, median, mode preserving ties, sample variance, standard deviation) across crop, soil, and weather sources.
- Isolates the controlled cross-sectional cohort: **Rice, Kharif, 2019 ($N=22$ states)**.
- Computes Pearson and Spearman correlation coefficients with crop yield.
- Performs a two-sided **Welch's independent two-sample $t$-test** comparing mean yields across rainfall regimes (median split at $1,312.03$ mm):
  - Result: $t = 3.398987$, $\text{df} = 17.474$, $p = 0.003308$ (Reject $H_0$ at $\alpha = 0.05$).
  - Lower-rainfall states show higher observed mean yields ($2.871$ vs. $1.963$ t/ha), largely attributable to high irrigation access in states like Punjab and Haryana.
- Outputs 12 summary CSVs and a detailed findings report to `results/statistical_analysis/`.

### Module 4: `regression.py` (Prathamesh Samel)
- Implements Multiple Linear Regression from first principles using the **Normal Equations**:
  $$(X^T X + \lambda I)\beta = X^T y \implies \beta = \text{scipy.linalg.solve}(X^T X + \lambda I, X^T y)$$
- Excludes `production` to eliminate direct target data leakage; models $\log(\text{yield})$ to normalize variance.
- Uses a time-based train/test split: **Train (1997–2015, $N=15,268$)** and **Test (2016–2019, $N=4,213$)**.
- Evaluates four models on unseen holdout test years:
  - **Baseline Benchmark (Crop Mean):** Test $R^2 = 0.7756$, $\text{RMSE} = 0.6881$.
  - **Model 1 (Factors Only):** Test $R^2 = 0.0018$, $\text{RMSE} = 1.4515$ (raw input factors alone explain $<1\%$ of variance).
  - **Model 2 (Factors + Crop Fixed Effects):** Test $R^2 = 0.7779$, $\text{RMSE} = 0.6847$ (controlling for crop variety substantially improves fit).
  - **Model 3 (Expanded Soil & Weather):** Test $R^2 = 0.7887$, $\text{RMSE} = 0.6678$.
- Confirms significant error reduction over Baseline via paired $t$-test ($p < 0.001$).
- Saves 3 focused figures in `output_graphs/regression/`.

### Module 5: `forecasting.py` (Prathamesh Samel)
- Implements time series analysis aligned with Experiment 12:
  - Formulates yearly median yield series for four staple crops (Rice, Wheat, Maize, Sugarcane) with Pandas `DatetimeIndex`.
  - Conducts Augmented Dickey-Fuller (ADF) testing; applies first-order differencing ($d=1$) to stabilize the mean.
  - Compares 3-year Simple Moving Average (SMA) and EWMA (span=3); demonstrates that EWMA tracks recent shocks with lower Mean Absolute Deviation.
  - Benchmarks candidate Statsmodels $\text{ARIMA}(p, d, q)$ specifications against a Last Value baseline across a 4-year holdout (2016–2019).
  - Refits optimal models on all data through 2019 and generates 5-year future yield projections (2020–2024) with 95% confidence intervals.
- Saves 4 focused figures in `output_graphs/forecasting/`.

---

## 8. Verification & Test Output

Running `python main.py` executes all five stages sequentially in ~16 seconds with zero errors:

```text
============================================================================
 CAPSTONE PROJECT EXECUTION SUMMARY
============================================================================
Step   Analytical Stage                       Author               Status     Runtime
----------------------------------------------------------------------------
Step 1  Data Cleaning & Exploration            Shatakshi Sankhe     SUCCESS    0.10s
Step 2  Data Visualizations (3 Dashboards)     Shatakshi Sankhe     SUCCESS    5.62s
Step 3  Statistical Analysis & Hypothesis Test Hitans Sangrajka    SUCCESS    1.65s
Step 4  Multiple Linear Regression & Diagnosti Prathamesh Samel     SUCCESS    3.40s
Step 5  Time Series & ARIMA Forecasting        Prathamesh Samel     SUCCESS    5.28s
----------------------------------------------------------------------------
Total Pipeline Runtime: 16.05 seconds.
All output graphs saved to: output_graphs/
All numerical summaries saved to: results/
============================================================================
```
