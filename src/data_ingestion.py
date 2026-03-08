"""
Data ingestion script.
Loads raw CSV and performs basic validation before passing to preprocessing.
"""
import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw_data(raw_path: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def validate_data(df: pd.DataFrame, target_col: str) -> None:
    assert target_col in df.columns, f"Target column '{target_col}' not found"
    assert df[target_col].nunique() == 2, "Target must be binary"
    logger.info(f"Churn rate: {df[target_col].mean():.2%}")
    logger.info("Data validation passed.")


if __name__ == "__main__":
    params = load_params()
    df = load_raw_data(params["data"]["raw_path"])
    validate_data(df, params["data"]["target_column"])
    logger.info("Ingestion complete.")
