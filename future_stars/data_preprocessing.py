import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import difflib


def preprocess_data(df, target_col="Future_Star"):

    if target_col not in df.columns:
        # helpful debugging information: show closest column name matches
        cols = list(df.columns)
        close = difflib.get_close_matches(target_col, cols, n=5, cutoff=0.5)
        msg = (
            f"Target column '{target_col}' not found in DataFrame."
            f" Available columns: {cols[:10]}... (showing 10)."
        )
        if close:
            msg += f" Did you mean one of: {close}?"
        raise KeyError(msg)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # keep building blocks for preprocessing available (not applied here)
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in ["Role"] if c in X.columns]

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    # Return raw X, y and the preprocessor — splitting is handled by the caller (main.py)
    return X, y, preprocessor
