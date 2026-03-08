"""
Training stage.
Reads feature data → trains XGBoost → logs to MLflow/DagsHub → saves model artifact.
DVC stage: train
"""
import pandas as pd
import yaml
import logging
import os
import json
import pickle
import mlflow
import mlflow.xgboost
import dagshub
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(target_col: str):
    train_df = pd.read_csv("data/features/train_features.csv")
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    return X_train, y_train


def build_model(train_params: dict) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=train_params["n_estimators"],
        max_depth=train_params["max_depth"],
        learning_rate=train_params["learning_rate"],
        random_state=train_params["random_state"],
        use_label_encoder=False,
        eval_metric="logloss",
    )


if __name__ == "__main__":
    params = load_params()
    target_col = params["data"]["target_column"]
    train_params = params["train"]
    mlflow_params = params["mlflow"]

    # Connect MLflow to DagsHub
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name="telecom_churn_mlops",
        mlflow=True,
    )
    mlflow.set_experiment(mlflow_params["experiment_name"])

    X_train, y_train = load_data(target_col)

    with mlflow.start_run(run_name="xgboost_churn"):
        # Enable autologging
        mlflow.xgboost.autolog()

        model = build_model(train_params)
        model.fit(X_train, y_train)

        # Train-set metrics
        y_pred_proba = model.predict_proba(X_train)[:, 1]
        y_pred = (y_pred_proba >= params["evaluate"]["threshold"]).astype(int)
        train_metrics = {
            "train_roc_auc": roc_auc_score(y_train, y_pred_proba),
            "train_f1": f1_score(y_train, y_pred),
            "train_precision": precision_score(y_train, y_pred),
            "train_recall": recall_score(y_train, y_pred),
        }
        mlflow.log_metrics(train_metrics)
        mlflow.log_params(train_params)
        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("dataset", "telecom_churn")

        # Log model with signature
        sig = infer_signature(X_train, model.predict(X_train))
        mlflow.xgboost.log_model(model, "model", signature=sig)

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")

        for k, v in train_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    # Save model artifact for DVC tracking
    os.makedirs("models", exist_ok=True)
    with open("models/TelecomChurnModel.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save train metrics for DVC
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)

    logger.info("Training complete. Model saved to models/TelecomChurnModel.pkl")
