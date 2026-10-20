import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename="crop_yield.csv"):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.lower()
    return df


def save_and_show(filename):
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def fit_line(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    return slope, intercept


def r_squared(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - ss_res / ss_tot


def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100
