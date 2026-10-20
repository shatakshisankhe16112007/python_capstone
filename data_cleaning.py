import pandas as pd

def load_dataset():
    df = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\CAPSTONE PROJECT\python_capstone\dataset\crop_yield.csv")
    return df

def explore_dataset(df):

    print("FIRST FIVE ROWS")
    print(df.head())

    print("\nDATASET SHAPE")
    print(df.shape)

    print("\nCOLUMN NAMES")
    print(df.columns.tolist())

    print("\nDATA TYPES")
    print(df.dtypes)

    print("\nMISSING VALUES")
    print(df.isnull().sum())

    print("\nDUPLICATE RECORDS")
    print(df.duplicated().sum())

    print("\nNUMERICAL SUMMARY")
    print(df.describe())


# Execute the program
if __name__ == "__main__":
    data = load_dataset()
    explore_dataset(data)