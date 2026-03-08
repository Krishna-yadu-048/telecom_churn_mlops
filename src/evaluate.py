"""
Evaluation stage.
Loads test features + saved model → computes metrics → saves JSON + plot CSVs.
DVC stage: evaluate
"""
import pandas as pd
import numpy as np
import yaml
import logging
import os
import json
import pickle
import mlflow
import dagshub
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve,
)
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    params = load_params()
    target_col = params["data"]["target_column"]
    threshold = params["evaluate"]["threshold"]

    test_df = pd.read_csv("data/features/test_features.csv")
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    with open("models/TelecomChurnModel.pkl", "rb") as f:
        model = pickle.load(f)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "test_roc_auc": roc_auc_score(y_test, y_pred_proba),
        "test_f1": f1_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
    }

    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    os.makedirs("metrics", exist_ok=True)

    # Save test metrics JSON (DVC-tracked)
    with open("metrics/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ROC curve CSV for DVC plots
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv("metrics/roc_curve.csv", index=False)

    # Confusion matrix CSV for DVC plots
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
    cm_df.to_csv("metrics/confusion_matrix.csv")

    # Log to MLflow
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name="telecom_churn_mlops",
        mlflow=True,
    )
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("metrics/roc_curve.csv")
        mlflow.log_artifact("metrics/confusion_matrix.csv")

    logger.info("Evaluation complete. Metrics saved to metrics/")
