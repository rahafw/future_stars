import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report
)

def evaluate(name, pipeline, X_test, y_test, threshold, out_dir="outputs"):
    proba = pipeline.predict_proba(X_test)[:,1]
    pred = (proba >= threshold).astype(int)

    prec, rec, _ = precision_recall_curve(y_test, proba)
    metrics = {
        "model": name,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(auc(rec, prec)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, digits=3, zero_division=0)
    }

    Path(out_dir).mkdir(exist_ok=True)
    with open(Path(out_dir)/"metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
