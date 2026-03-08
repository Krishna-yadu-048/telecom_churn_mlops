"""
Batch prediction utility.
Loads Production model from MLflow registry → scores a CSV → outputs predictions.
Usage: python src/predict.py --input data/raw/new_customers.csv --output predictions.csv
"""
import argparse
import pandas as pd
import mlflow.pyfunc
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_production_model(model_name: str):
    model_uri = f"models:/{model_name}/Production"
    logger.info(f"Loading model from: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


def predict(model, input_path: str, output_path: str, threshold: float = 0.5) -> None:
    df = pd.read_csv(input_path)
    logger.info(f"Scoring {len(df)} records...")
    scores = model.predict(df)
    result_df = df.copy()
    result_df["churn_score"] = scores
    result_df["churn_prediction"] = (scores >= threshold).astype(int)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch churn prediction")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--model-name", default="TelecomChurnModel")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model = load_production_model(args.model_name)
    predict(model, args.input, args.output, args.threshold)
