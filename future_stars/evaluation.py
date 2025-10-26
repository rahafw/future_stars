import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, mean_absolute_error,
    mean_squared_error
)

def evaluate_model_with_metrics(name, model, X_train, y_train, X_test, y_test, preprocessor, threshold=0.8):
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)

    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "F1 Score": round(f1_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "MAE": round(mean_absolute_error(y_test, y_pred_prob), 4),
        "MSE": round(mean_squared_error(y_test, y_pred_prob), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_prob)), 4)
    }
