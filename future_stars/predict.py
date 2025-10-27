import argparse
import pandas as pd
from pathlib import Path
from .model import load_model
from .preprocessing import (
    clean_data, normalize_country, normalize_positions, build_per90_features
)

# Features the model was trained on
ALLOWED_FEATURES = [
    "GA_per90", "xGA_per90", "Def_Actions_per90", "ProgPass_per90",
    "Save_Pct", "Saves_per90", "Role"
]

def get_key_metric(row):
    """Return the role-specific key metric for scouts, with clear units."""
    if row["Pos"] == "GK":
        if pd.notna(row.get("Save_Pct")):
            return f"Save%: {(row['Save_Pct']*100):.2f}%"
        elif pd.notna(row.get("Saves_per90")):
            return f"Saves90: {row['Saves_per90']:.2f} saves/90"
        else:
            return "No GK metric"
    elif row["Pos"] == "DF":
        return f"DefActions90: {row.get('Def_Actions_per90', 0):.2f} actions/90"
    elif row["Pos"] in ["MF", "FW"]:
        ga = row.get("GA_per90")
        xga = row.get("xGA_per90")
        if pd.notna(ga) and (pd.isna(xga) or ga >= xga):
            return f"GA90: {ga:.2f} contrib/90"
        elif pd.notna(xga):
            return f"xGA90: {xga:.2f} expected contrib/90"
        else:
            return "No attacking metric"
    return "NA"



def predict(data=None, data_path=None, model_path="model/model.pkl", save_path=None):
    """
    Run predictions on a DataFrame or CSV and return results with:
    Player Name, Age, Pos, Prediction, Probability, Key Metric
    """

    # Load trained model
    model = load_model(model_path)

    # Handle input
    if data is not None:
        df = data.copy()
    elif data_path is not None:
        df = pd.read_csv(data_path)
    else:
        raise ValueError("You must provide either a DataFrame (data) or a CSV path (data_path).")

    # Preprocess
    df = clean_data(df)
    df = normalize_country(df)
    df = normalize_positions(df)
    df = build_per90_features(df)

    # Select features for model
    avail = [c for c in ALLOWED_FEATURES if c in df.columns]
    df_model = df[avail].copy()

    # Predictions
    preds = model.predict(df_model)
    proba = model.predict_proba(df_model)[:, 1]

    # Attach results
    results = df.copy()
    results["Prediction"] = pd.Series(preds).map({1: "Future Star", 0: "Not Future Star"})
    results["Probability"] = (proba * 100).round(2).astype(str) + "%"
    results["Key Metric"] = results.apply(get_key_metric, axis=1)

    # Final output for scouts
    final_results = results[["Player", "Age", "Pos", "Prediction", "Probability", "Key Metric"]]

    # Save on output file to track testing
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        final_results.to_csv(save_path, index=False)
        print(f"Predictions saved to {save_path}")

    return final_results


# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="test.csv", help="CSV to predict on")
    parser.add_argument("--model", type=str, default="model/model.pkl", help="Path to trained model.pkl")
    parser.add_argument("--save", type=str, default="outputs/predictions.csv", help="Where to save results")
    args = parser.parse_args()

    predict(data_path=args.data, model_path=args.model, save_path=args.save)
