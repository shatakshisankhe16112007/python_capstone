import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common_utils import load_data, save_and_show, fit_line, r_squared, rmse, mae

factors = ["log_area", "fert_per_ha", "pest_per_ha"]
category_columns = ["crop", "season", "state"]
train_end_year = 2015


def prepare_data(df):
    df = df[df["year"] <= 2019]
    df = df[(df["yield"] > 0) & (df["area"] > 0)].copy()
    df["fert_per_ha"] = df["fertilizer"] / df["area"]
    df["pest_per_ha"] = df["pesticide"] / df["area"]
    df["log_area"] = np.log(df["area"])
    df["log_yield"] = np.log(df["yield"])

    crop_mean = df.groupby("crop")["log_yield"].mean()
    crop_std = df.groupby("crop")["log_yield"].std()
    z_score = (df["log_yield"] - df["crop"].map(crop_mean)) / df["crop"].map(crop_std)
    df = df[(z_score > -4) & (z_score < 4)]
    return df.reset_index(drop=True)


def subtract_crop_average(df, column):
    averages = df.groupby("crop")[column].mean()
    return df[column] - df["crop"].map(averages)


def relationship_table(df):
    y = subtract_crop_average(df, "log_yield")
    rows = []
    for factor in factors:
        x = subtract_crop_average(df, factor)
        slope, intercept = fit_line(x.to_numpy(), y.to_numpy())
        rows.append({
            "factor": factor,
            "corr_all_crops": round(df[factor].corr(df["log_yield"]), 3),
            "corr_within_crop": round(x.corr(y), 3),
            "slope": round(slope, 5),
            "yield_change_pct_per_1_std": round((np.exp(slope * x.std()) - 1) * 100, 2),
        })
    return pd.DataFrame(rows)


def get_levels(train):
    levels = {}
    for col in category_columns:
        names = sorted(train[col].unique())
        levels[col] = names[1:]
    return levels


def build_matrix(df, use_categories, levels):
    columns = [np.ones(len(df))]
    names = ["intercept"]
    for factor in factors:
        columns.append(df[factor].to_numpy())
        names.append(factor)
    if use_categories:
        for col in category_columns:
            for level in levels[col]:
                columns.append(np.where(df[col] == level, 1.0, 0.0))
                names.append(col + "=" + str(level))
    return np.column_stack(columns), names


def fit_model(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def crop_average_baseline(train, test):
    crop_mean = train.groupby("crop")["log_yield"].mean()
    predicted = test["crop"].map(crop_mean)
    predicted = predicted.fillna(train["log_yield"].mean())
    return predicted.to_numpy()


def score_row(model_name, split_name, actual, predicted):
    return {
        "model": model_name,
        "split": split_name,
        "R2": round(r_squared(actual, predicted), 4),
        "RMSE": round(rmse(actual, predicted), 4),
        "MAE": round(mae(actual, predicted), 4),
    }


def run_models(train, test):
    y_train = train["log_yield"].to_numpy()
    y_test = test["log_yield"].to_numpy()
    levels = get_levels(train)

    rows = []
    results = {}

    model_names = ["Model 1: factors only", "Model 2: factors + crop, season, state"]
    use_categories_list = [False, True]
    for i in range(len(model_names)):
        X_train, names = build_matrix(train, use_categories_list[i], levels)
        X_test, names = build_matrix(test, use_categories_list[i], levels)
        beta = fit_model(X_train, y_train)
        predicted_train = X_train @ beta
        predicted_test = X_test @ beta
        rows.append(score_row(model_names[i], "train", y_train, predicted_train))
        rows.append(score_row(model_names[i], "test", y_test, predicted_test))
        results[model_names[i]] = {"beta": beta, "names": names, "predicted_test": predicted_test}

    rows.append(score_row("Baseline: crop average", "train", y_train, crop_average_baseline(train, train)))
    rows.append(score_row("Baseline: crop average", "test", y_test, crop_average_baseline(train, test)))
    return pd.DataFrame(rows), results


def coefficient_table(result):
    rows = []
    for i in range(len(factors) + 1):
        rows.append({"term": result["names"][i], "coefficient": round(result["beta"][i], 6)})
    return pd.DataFrame(rows)


def plot_correlations(df):
    demeaned = pd.DataFrame()
    for col in factors + ["log_yield"]:
        demeaned[col] = subtract_crop_average(df, col)

    plt.figure(figsize=(7, 5))
    sns.heatmap(demeaned.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation of factors with log(yield), crop average removed")
    save_and_show("reg_1_correlation_heatmap.png")


def plot_relationships(df):
    demeaned = pd.DataFrame()
    for col in factors + ["log_yield"]:
        demeaned[col] = subtract_crop_average(df, col)
    sample = demeaned.sample(3000, random_state=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(len(factors)):
        sns.regplot(data=sample, x=factors[i], y="log_yield", ax=axes[i],
                    scatter_kws={"alpha": 0.15, "s": 8}, line_kws={"color": "red"})
        axes[i].set_title("log(yield) vs " + factors[i])
        axes[i].set_xlabel(factors[i] + " (crop average removed)")
        axes[i].set_ylabel("log(yield) (crop average removed)")
    save_and_show("reg_2_factor_relationships.png")


def plot_model_scores(scores):
    test_scores = scores[scores["split"] == "test"]
    plt.figure(figsize=(9, 5))
    plt.bar(test_scores["model"], test_scores["R2"], color=["gray", "orange", "green"])
    for i in range(len(test_scores)):
        plt.text(i, test_scores["R2"].iloc[i] + 0.01, str(test_scores["R2"].iloc[i]), ha="center")
    plt.xlabel("Model")
    plt.ylabel("R2 on test years (2016-2019)")
    plt.title("How well each model explains log(yield) on unseen years")
    plt.xticks(rotation=10)
    save_and_show("reg_3_model_scores.png")


def plot_actual_vs_predicted(test, predicted):
    actual = test["log_yield"].to_numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, alpha=0.25, s=10)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], "r--", label="Perfect prediction")
    plt.xlabel("Actual log(yield)")
    plt.ylabel("Predicted log(yield)")
    plt.title("Model 2: actual vs predicted (test years)")
    plt.legend()
    save_and_show("reg_4_actual_vs_predicted.png")


def plot_residuals(test, predicted):
    residuals = test["log_yield"].to_numpy() - predicted
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(residuals, bins=40, kde=True, ax=axes[0])
    axes[0].set_xlabel("Residual (actual - predicted)")
    axes[0].set_title("Distribution of residuals")
    axes[1].scatter(predicted, residuals, alpha=0.25, s=10)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted log(yield)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs predicted")
    save_and_show("reg_5_residuals.png")


def print_limitations():
    print("\nLIMITATIONS")
    print("1. The dataset has no rainfall, temperature or soil columns, so weather and soil cannot be modelled.")
    print("2. Fertilizer and pesticide per hectare have the same value for every row in a year,")
    print("   so they behave like a trend over time and not like inputs of a single farm.")
    print("3. The model shows association, not cause and effect.")
    print("4. Yield of coconut is in the thousands while other crops are 1 to 3, so log(yield) and crop effects are used.")
    print("5. The yield column does not equal production / area for most rows; the yield column is used as given.")
    print("6. Test years (2016-2019) are close to the training years, so new states or later years may not fit as well.")


def main():
    sns.set_style("whitegrid")

    df = prepare_data(load_data())
    train = df[df["year"] <= train_end_year]
    test = df[df["year"] > train_end_year]
    print("Rows used:", len(df))
    print("Training rows (1997-2015):", len(train))
    print("Test rows (2016-2019):", len(test))

    print("\nRELATIONSHIP BETWEEN YIELD AND FACTORS")
    print(relationship_table(df).to_string(index=False))

    scores, results = run_models(train, test)
    print("\nMODEL EVALUATION")
    print(scores.to_string(index=False))

    print("\nCOEFFICIENTS OF AGRICULTURAL FACTORS")
    for name in results:
        print(name)
        print(coefficient_table(results[name]).to_string(index=False))

    plot_correlations(df)
    plot_relationships(df)
    plot_model_scores(scores)
    best = results["Model 2: factors + crop, season, state"]
    plot_actual_vs_predicted(test, best["predicted_test"])
    plot_residuals(test, best["predicted_test"])

    print_limitations()


if __name__ == "__main__":
    main()
