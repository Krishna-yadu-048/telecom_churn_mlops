"""
Preprocessing stage.
Reads raw data → cleans → splits into train/test → saves to data/processed/.
DVC stage: preprocess
"""
import pandas as pd
import yaml
import logging
from sklearn.model_selection import train_test_split
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def preprocess(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    data_params = params["data"]
    pre_params = params["preprocessing"]

    # Drop specified columns
    drop_cols = pre_params.get("drop_columns", [])
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
        logger.info(f"Dropped columns: {drop_cols}")

    # Handle missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"Found {missing} missing values — filling with median/mode")
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ["float64", "int64"]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)

    logger.info(f"Preprocessed shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame, target_col: str, test_size: float, random_state: int):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    params = load_params()
    data_params = params["data"]

    df = pd.read_csv(data_params["raw_path"])
    df = preprocess(df, params)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_col=data_params["target_column"],
        test_size=data_params["test_size"],
        random_state=data_params["random_state"],
    )

    os.makedirs("data/processed", exist_ok=True)
    train_df = X_train.copy()
    train_df[data_params["target_column"]] = y_train.values
    test_df = X_test.copy()
    test_df[data_params["target_column"]] = y_test.values

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    logger.info("Saved train.csv and test.csv to data/processed/")
