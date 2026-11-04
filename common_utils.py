"""common_utils.py - Shared utilities for Agricultural Yield Analysis.
Author: Prathamesh Samel (Python Programming Laboratory)
Provides dataset loading, table merging, metric calculations,
and the Normal Equations linear algebra solver via SciPy (Exp 6, 7, 8, 11).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output_graphs"
RESULTS_DIR = BASE_DIR / "results"


def ensure_directories():
    """Create all required output directories if they do not exist."""
    for sub in ["visualization", "statistical_analysis", "regression", "forecasting"]:
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)


def load_dataset(filename="crop_yield.csv"):
    """Load a CSV dataset, strip whitespace, and normalize column names."""
    df = pd.read_csv(DATA_DIR / filename)
    df.columns = df.columns.str.strip().str.lower()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def load_all_datasets():
    """Load crop yield, soil chemistry, and weather datasets."""
    return load_dataset("crop_yield.csv"), load_dataset("state_soil_data.csv"), load_dataset("state_weather_data_1997_2020.csv")


def merge_agricultural_data(crop=None, soil=None, weather=None):
    """Merge crop records with weather (state, year) and soil (state) Many-to-One."""
    if crop is None or soil is None or weather is None:
        crop, soil, weather = load_all_datasets()
    merged = crop.merge(weather, on=["state", "year"], how="left", validate="many_to_one")
    return merged.merge(soil, on="state", how="left", validate="many_to_one")


def solve_normal_equations(X, y, ridge_alpha=1e-4):
    """Solve (X^T X + lambda*I) beta = X^T y via scipy.linalg.solve (Exp 11)."""
    X_mat, y_vec = np.asarray(X, float), np.asarray(y, float)
    XtX = X_mat.T @ X_mat
    Xty = X_mat.T @ y_vec
    np.fill_diagonal(XtX, XtX.diagonal() + ridge_alpha)
    try:
        return linalg.solve(XtX, Xty, assume_a="sym")
    except linalg.LinAlgError:
        return linalg.pinv(XtX) @ Xty


def r_squared(actual, predicted):
    """Compute R-squared goodness of fit."""
    ss_tot = np.sum((np.asarray(actual, float) - np.mean(actual)) ** 2)
    return float(1.0 - np.sum((np.asarray(actual, float) - np.asarray(predicted, float)) ** 2) / ss_tot) if ss_tot != 0 else 0.0


def rmse(actual, predicted):
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(actual, float) - np.asarray(predicted, float)) ** 2)))


def mae(actual, predicted):
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(predicted, float))))


def fit_line(x, y):
    """Compute simple slope and intercept via least squares formula (Exp 7)."""
    x_arr, y_arr = np.asarray(x, float), np.asarray(y, float)
    denom = np.sum((x_arr - np.mean(x_arr)) ** 2)
    slope = np.sum((x_arr - np.mean(x_arr)) * (y_arr - np.mean(y_arr))) / denom if denom != 0 else 0.0
    return float(slope), float(np.mean(y_arr) - slope * np.mean(x_arr))


def save_plot(filename, output_subfolder="regression", dpi=300):
    """Save the current matplotlib figure and close it cleanly."""
    out = OUTPUT_DIR / output_subfolder / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
