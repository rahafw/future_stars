# preprocessing.py
import pandas as pd
import numpy as np
import pycountry

SELECTED_COLUMNS = [
    # Player info
    "Player", "Nation", "Pos", "Age",

    # Playing time
    "MP", "Starts", "Min", "90s",

    # Attacking
    "Gls", "Ast", "xG", "xAG", "G+A",

    # Defensive
    "Tkl", "TklW", "Blocks_stats_defense", "Clr", "Err",

    # Passing & Creativity Stats
    "PrgP", "PrgC", "KP", "xA",

    # Goalkeeping
    "GA", "Saves", "Save%", "CS", "CS%", "PKA", "PKsv",

    # Miscellaneous Stats
    "CrdY", "CrdR"
]

# Validation & Cleaning
def validate_columns(df: pd.DataFrame):
    """Ensure DataFrame has all required columns."""
    missing = [col for col in SELECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[SELECTED_COLUMNS]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()

    if "Min" in df.columns:
        df = df[df["Min"] > 0]

    if "Pos" in df.columns:
        df["Pos"] = df["Pos"].fillna("Unknown")

    return df.reset_index(drop=True)


def normalize_country(df: pd.DataFrame) -> pd.DataFrame:
    if "Nation" not in df.columns:
        return df

    def full_country_name(alpha3):
        try:
            m = pycountry.countries.get(alpha_3=str(alpha3))
            return m.name if m else alpha3
        except Exception:
            return alpha3

    df = df.copy()
    df["Nation"] = df["Nation"].astype(str).str.split().str[-1].apply(full_country_name)
    return df

def map_role(pos: str) -> str:
    pos = str(pos)
    if "GK" in pos: return "GK"
    if "DF" in pos: return "DF"
    if "MF" in pos: return "MF"
    if "FW" in pos: return "FW"
    return "Other"

def normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    if "Pos" not in df.columns:
        return df
    df = df.copy()
    df["Role"] = df["Pos"].apply(map_role)
    return df


# Feature engineering
def safe_per90(num, minutes):
    return (num / minutes.replace(0, np.nan)) * 90

def build_per90_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Contributions
    if {"G+A", "Min"}.issubset(df.columns):
        df["GA_per90"] = safe_per90(df["G+A"], df["Min"])
    if {"xG", "xAG", "Min"}.issubset(df.columns):
        df["xGA_per90"] = safe_per90(df["xG"] + df["xAG"], df["Min"])

    # Defense
    comb_cols = [c for c in ["Tkl", "Blocks_stats_defense", "Clr"] if c in df.columns]
    if comb_cols and "Min" in df.columns:
        df["Def_Actions_per90"] = safe_per90(df[comb_cols].sum(axis=1), df["Min"])

    # Progression
    if {"PrgP", "Min"}.issubset(df.columns):
        df["ProgPass_per90"] = safe_per90(df["PrgP"], df["Min"])

    # GK save %
    if "Save%" in df.columns:
        sp_raw = df["Save%"].astype(str).str.strip().str.replace("%", "", regex=False)
        sp = pd.to_numeric(sp_raw, errors="coerce")
        df["Save_Pct"] = np.where(sp > 1.0, sp / 100.0, sp)
    else:
        df["Save_Pct"] = np.nan

    return df

# Labeling
def role_score_row(row):
    role = row.get("Role", "Other")
    if role == "GK":
        if pd.notna(row.get("Save_Pct")):
            return row.get("Save_Pct")
        return row.get("Saves_per90")
    elif role == "DF":
        return row.get("Def_Actions_per90")
    elif role in ("MF", "FW"):
        ga = row.get("GA_per90")
        xga = row.get("xGA_per90")
        if pd.isna(ga) and pd.isna(xga):
            return np.nan
        if pd.isna(ga): return xga
        if pd.isna(xga): return ga
        return max(ga, xga)
    return np.nan

def apply_future_star_label(df: pd.DataFrame, pct=0.80):
    """
    Compute role-specific score, find thresholds, assign Future_Star label.
    """
    if "Role" not in df.columns:
        raise ValueError("Required column 'Role' not found in dataset")

    df = df.copy()
    df["Role_Score"] = df.apply(role_score_row, axis=1)

    thresholds = {}
    for role, sub in df.groupby("Role"):
        valid = sub["Role_Score"].dropna()
        thresholds[role] = valid.quantile(pct) if not valid.empty else np.nan

    df["role_threshold"] = df["Role"].map(thresholds)
    df["Future_Star"] = (df["Role_Score"] >= df["role_threshold"]).astype(int)

    return df, thresholds

def full_preprocessing(df: pd.DataFrame):
    df = validate_columns(df)
    df = clean_data(df)
    df = normalize_country(df)
    df = normalize_positions(df)
    df = build_per90_features(df)
    df, thresholds = apply_future_star_label(df)
    return df, thresholds
