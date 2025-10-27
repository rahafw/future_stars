import pickle
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Model building utilities
def build_xgb(scale_pos_weight=1.0):
    """
    Build an XGBoost classifier with tuned hyperparameters.
    """
    return XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        subsample=1.0,
        reg_lambda=1,
        reg_alpha=0.01,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )

def build_preprocessor(num_cols, cat_cols):
    """
    Build preprocessing transformer for numeric and categorical columns.
    """
    numeric = ("num", SimpleImputer(strategy="median"), num_cols)
    categorical = ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    return ColumnTransformer([numeric, categorical])

def build_pipeline(preprocessor, model):
    """
    Combine preprocessing and model into a pipeline.
    """
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])


# Save & Load
def save_model(pipeline, path="model/model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)

def load_model(path="model/model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
