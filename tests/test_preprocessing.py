"""Unit tests for preprocessing."""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess, split_data

SAMPLE_PARAMS = {
    "data": {
        "raw_path": "data/raw/telecom_churn.csv",
        "test_size": 0.2,
        "random_state": 42,
        "target_column": "Churn",
    },
    "preprocessing": {
        "drop_columns": [],
        "scale": "standard",
    },
}


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Churn": [0, 1, 0, 1, 0],
            "AccountWeeks": [128, 107, 137, 84, 75],
            "ContractRenewal": [1, 1, 1, 0, 1],
            "DataPlan": [1, 1, 0, 0, 1],
            "DataUsage": [2.7, 3.7, 0.0, 0.0, 1.5],
            "CustServCalls": [1, 1, 0, 2, 3],
            "DayMins": [265.1, 161.6, 243.4, 299.4, 150.0],
            "DayCalls": [110, 123, 114, 71, 95],
            "MonthlyCharge": [89.0, 82.0, 52.0, 57.0, 70.0],
            "OverageFee": [9.87, 9.78, 6.06, 3.1, 5.0],
            "RoamMins": [10.0, 13.7, 12.2, 6.6, 8.0],
        }
    )


def test_preprocess_no_drop(sample_df):
    result = preprocess(sample_df, SAMPLE_PARAMS)
    assert "Churn" in result.columns
    assert result.shape[0] == 5


def test_preprocess_drops_columns(sample_df):
    params = {**SAMPLE_PARAMS, "preprocessing": {"drop_columns": ["RoamMins"], "scale": "standard"}}
    result = preprocess(sample_df, params)
    assert "RoamMins" not in result.columns


def test_preprocess_handles_missing(sample_df):
    sample_df.loc[0, "DataUsage"] = None
    result = preprocess(sample_df, SAMPLE_PARAMS)
    assert result["DataUsage"].isnull().sum() == 0


def test_split_data_sizes(sample_df):
    X_train, X_test, y_train, y_test = split_data(
        sample_df, target_col="Churn", test_size=0.4, random_state=42
    )
    assert len(X_train) + len(X_test) == len(sample_df)
    assert "Churn" not in X_train.columns


def test_split_data_target_not_in_features(sample_df):
    X_train, X_test, y_train, y_test = split_data(
        sample_df, target_col="Churn", test_size=0.4, random_state=42
    )
    assert "Churn" not in X_train.columns
    assert "Churn" not in X_test.columns
