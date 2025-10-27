import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from .preprocessing import (
    clean_data,
    normalize_country,
    normalize_positions,
    build_per90_features,
    apply_future_star_label,
)
from .model import build_xgb, build_preprocessor, build_pipeline, save_model
from .evaluation import evaluate


def prepare_data(path, threshold=0.80):
    df = pd.read_csv(path)

    # Cleaning & preprocessing
    df = clean_data(df)
    df = normalize_country(df)
    df = normalize_positions(df)
    df = build_per90_features(df)

    # Labeling
    df, thresholds = apply_future_star_label(df, pct=threshold)

    # Features allowed for training
    allowed_features = [
        "GA_per90",
        "xGA_per90",
        "Def_Actions_per90",
        "ProgPass_per90",
        "Save_Pct",
        "Role"
    ]

    # Drop unused columns
    drop_cols = ["Player", "Nation", "Pos", "Role_Score", "role_threshold", "Future_Star"]
    X_full = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Keep only allowed features that exist in the dataset
    X = X_full[[c for c in allowed_features if c in X_full.columns]].copy()
    y = df["Future_Star"]

    # Separate categorical & numerical features
    num_cols = [c for c in X.columns if X[c].dtype != "object"]
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]

    print(f"Using features: {num_cols + cat_cols}")

    return X, y, num_cols, cat_cols, thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/players_data_light-2024_2025.csv")
    ap.add_argument("--threshold", type=float, default=0.80)
    args = ap.parse_args()

    # Prepare dataset
    X, y, num_cols, cat_cols, thresholds = prepare_data(args.data, threshold=args.threshold)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle imbalance
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    # Build model pipeline
    model = build_xgb(scale_pos_weight=scale_pos_weight)
    preproc = build_preprocessor(num_cols, cat_cols)
    pipe = build_pipeline(preproc, model)

    # Train model
    pipe.fit(X_train, y_train)

    # Evaluate performance
    metrics = evaluate("XGBoost", pipe, X_test, y_test, args.threshold)

    # Save trained pipeline
    save_model(pipe, "model/model.pkl")

    print(json.dumps(metrics, indent=2))
    print("Model trained, evaluated, and saved to model/model.pkl")


if __name__ == "__main__":
    main()
