import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\CAPSTONE PROJECT\python_capstone\dataset\crop_yield.csv")

df["yield"] = pd.to_numeric(df["yield"], errors="coerce")

yield_data = df["yield"].dropna()

print("STATISTICAL ANALYSIS OF CROP YIELD")

print("\nMean:", yield_data.mean())

print("Median:", yield_data.median())

print("Mode:")
print(yield_data.mode())

print("Variance:", yield_data.var())

print("Standard Deviation:", yield_data.std())

print("\nMinimum Yield:", yield_data.min())

print("Maximum Yield:", yield_data.max())

print("\nCORRELATION BETWEEN NUMERICAL VARIABLES")

numeric_data = df.select_dtypes(include=np.number)

print(numeric_data.corr())