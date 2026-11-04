"""main.py - Master Pipeline Execution File for Agricultural Yield Analysis.
Group 5: Shatakshi Sankhe, Hitans Sangrajka, Prathamesh Samel
Institution: K. J. Somaiya School of Engineering, Somaiya Vidyavihar University
Sequentially coordinates all five project stages across our laboratory syllabus.
"""

import os
import sys
import time
import shutil
from pathlib import Path

# Configure matplotlib for headless execution
import matplotlib
matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common_utils import ensure_directories
import data_cleaning
import statistical_analysis
from regression import run_regression_analysis
from forecasting import run_forecasting_analysis


def run_shatakshi_cleaning():
    """Execute Shatakshi Sankhe's data cleaning and exploration."""
    df = data_cleaning.load_dataset()
    data_cleaning.explore_dataset(df)


def run_shatakshi_visualization():
    """Execute Shatakshi Sankhe's 3 four-panel dashboard visualizations."""
    import runpy
    runpy.run_path(str(BASE_DIR / "visualization.py"), run_name="__main__")
    # Organize outputs into the agreed subfolder
    sub_dir = BASE_DIR / "output_graphs" / "visualization"
    sub_dir.mkdir(parents=True, exist_ok=True)
    dashboards = [
        "figure1_regional_production.png",
        "figure2_agricultural_relationships.png",
        "figure3_yield_seasonal_analysis.png",
    ]
    for name in dashboards:
        src = BASE_DIR / "output_graphs" / name
        if src.exists():
            shutil.copy2(src, sub_dir / name)


def execute_pipeline():
    """Execute all five project stages sequentially with runtime tracking."""
    start_time = time.time()
    ensure_directories()

    stages = [
        (1, "Data Cleaning & Exploration", "Shatakshi Sankhe", run_shatakshi_cleaning),
        (2, "Data Visualizations (3 Dashboards)", "Shatakshi Sankhe", run_shatakshi_visualization),
        (3, "Statistical Analysis & Hypothesis Testing", "Hitans Sangrajka", statistical_analysis.run_analysis),
        (4, "Multiple Linear Regression & Diagnostics", "Prathamesh Samel", run_regression_analysis),
        (5, "Time Series & ARIMA Forecasting", "Prathamesh Samel", run_forecasting_analysis),
    ]

    summary = []
    for step, name, author, func in stages:
        print("\n" + "=" * 76)
        print(f" STEP {step}: {name.upper()} [{author}]")
        print("=" * 76)
        t0 = time.time()
        try:
            func()
            elapsed = time.time() - t0
            summary.append((step, name, author, "SUCCESS", f"{elapsed:.2f}s"))
            print(f"---> Step {step} completed in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - t0
            summary.append((step, name, author, f"FAILED: {e}", f"{elapsed:.2f}s"))
            print(f"[ERROR] Step {step} failed: {e}", file=sys.stderr)
            return False

    total_time = time.time() - start_time
    print("\n" + "=" * 76)
    print(" CAPSTONE PROJECT EXECUTION SUMMARY")
    print("=" * 76)
    print(f"{'Step':<6} {'Analytical Stage':<38} {'Author':<20} {'Status':<10} {'Runtime'}")
    print("-" * 76)
    for num, name, author, status, duration in summary:
        print(f"Step {num:<2} {name:<38} {author:<20} {status:<10} {duration}")
    print("-" * 76)
    print(f"Total Pipeline Runtime: {total_time:.2f} seconds.")
    print("All output graphs saved to: output_graphs/")
    print("All numerical summaries saved to: results/")
    print("=" * 76)
    return True


if __name__ == "__main__":
    if not execute_pipeline():
        sys.exit(1)
