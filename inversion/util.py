import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_dataset(df):
    scaler = StandardScaler()
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df), columns=["x", "y", "z"])
    df = df.sort_values(by="x")
    return df


def sort_2D_test_data(X_test, y_test):
    df_a = pd.DataFrame(X_test, columns=["x"])
    df_a["y"] = y_test
    df_a = df_a.sort_values(by="x")
    X_test = df_a[["x"]]
    y_test = df_a["y"]
    return X_test, y_test
