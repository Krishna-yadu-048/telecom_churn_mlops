"""
Feature engineering stage.
Reads processed train/test → engineers features → scales → saves to data/features/.
DVC stage: feature_engineering
"""
import pandas as pd
import numpy as np
import yaml
import logging
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def engineer_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Add domain-relevant derived features for churn prediction."""
    df = df.copy()

    # Revenue-related features
    df["ChargePerMin"] = df["MonthlyCharge"] / (df["DayMins"] + 1e-6)
    df["OverageRatio"] = df["OverageFee"] / (df["MonthlyCharge"] + 1e-6)
    df["RoamToDay"] = df["RoamMins"] / (df["DayMins"] + 1e-6)

    # Engagement features
    df["CallsPerDay"] = df["DayCalls"] / (df["AccountWeeks"] / 4 + 1e-6)
    df["HighServiceCalls"] = (df["CustServCalls"] >= 3).astype(int)

    logger.info(f"Feature engineered shape: {df.shape}")
    return df


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    scale_method: str,
) -> tuple:
    """Fit scaler on train, transform both sets. Save scaler for inference."""
    feature_cols = [c for c in train_df.columns if c != target_col]

    if scale_method == "standard":
        scaler = StandardScaler()
    elif scale_method == "minmax":
        scaler = MinMaxScaler()
    else:
        logger.info("No scaling applied.")
        return train_df, test_df, None

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler ({scale_method}) fitted and saved to models/scaler.pkl")
    return train_df, test_df, scaler


if __name__ == "__main__":
    params = load_params()
    target_col = params["data"]["target_column"]

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    train_df = engineer_features(train_df, target_col)
    test_df = engineer_features(test_df, target_col)

    train_df, test_df, _ = scale_features(
        train_df, test_df, target_col, params["preprocessing"]["scale"]
    )

    os.makedirs("data/features", exist_ok=True)
    train_df.to_csv("data/features/train_features.csv", index=False)
    test_df.to_csv("data/features/test_features.csv", index=False)
    logger.info("Saved train_features.csv and test_features.csv to data/features/")
