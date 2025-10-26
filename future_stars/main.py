import pandas as pd
from sklearn.model_selection import train_test_split
from future_stars.data_preprocessing import preprocess_data
from future_stars.modeling import get_models
from future_stars.evaluation import evaluate_model_with_metrics


def main():
    print("🚀 Running Future Stars pipeline...")

    data_path = "/home/razan/code/future_stars/data/players_data_light-2024_2025.csv"
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # If the expected target column is missing, create a safe synthetic
    # label for quick testing (prefer using 'Gls' if available).
    if "Future_Star" not in df.columns:
        if "Gls" in df.columns:
            # mark players with >=5 goals as future stars (simple heuristic)
            df["Future_Star"] = (pd.to_numeric(df["Gls"], errors="coerce").fillna(0) >= 5).astype(int)
            print(
                f"⚠️ 'Future_Star' column missing — created synthetic target from 'Gls' with threshold >=5 (positives={df['Future_Star'].sum()})."
            )
        else:
            # fallback: create an all-zero target so pipeline can run
            df["Future_Star"] = 0
            print("⚠️ 'Future_Star' column missing and 'Gls' not found — created all-zero synthetic target.")

    X, y, preprocessor = preprocess_data(df)
    print(f"✅ Preprocessing completed: {X.shape[0]} samples ready for training")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("✅ Data split into training and test sets")

    rf, xgb_clf, lgbm_clf = get_models(y_train)
    print("✅ Models initialized successfully")

    results = []
    results.append(evaluate_model_with_metrics("RandomForest", rf, X_train, y_train, X_test, y_test, preprocessor))
    results.append(evaluate_model_with_metrics("XGBoost", xgb_clf, X_train, y_train, X_test, y_test, preprocessor))
    results.append(evaluate_model_with_metrics("LightGBM", lgbm_clf, X_train, y_train, X_test, y_test, preprocessor))

    print("\n📊 Model Evaluation Results:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
