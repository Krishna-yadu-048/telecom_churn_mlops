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
import dagshub.auth
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv

load_dotenv(override=True)
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
        eval_metric="logloss",
    )


if __name__ == "__main__":
    params = load_params()
    target_col = params["data"]["target_column"]
    train_params = params["train"]
    mlflow_params = params["mlflow"]

    # Authenticate with DagsHub token explicitly (required in Docker / CI)
    dagshub.auth.add_app_token(os.getenv("DAGSHUB_TOKEN"))

    # Connect MLflow to DagsHub
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name="telecom_churn_mlops",
        mlflow=True,
    )

    # Re-set credentials explicitly after dagshub.init()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

    mlflow.set_experiment(mlflow_params["experiment_name"])

    # Disable autolog entirely — we log everything manually
    mlflow.xgboost.autolog(disable=True)

    X_train, y_train = load_data(target_col)

    with mlflow.start_run(run_name="xgboost_churn"):

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

        # Log model explicitly with signature
        sig = infer_signature(X_train, model.predict_proba(X_train))
        mlflow.xgboost.log_model(model, artifact_path="model", signature=sig)
        logger.info("Model logged to MLflow.")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")

        for k, v in train_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    # Save model artifact locally for DVC tracking
    os.makedirs("models", exist_ok=True)
    with open("models/TelecomChurnModel.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save train metrics for DVC
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)

    logger.info("Training complete. Model saved to models/TelecomChurnModel.pkl")