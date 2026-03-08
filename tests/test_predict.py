"""Unit tests for prediction logic."""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class MockModel:
    """Minimal mock for mlflow pyfunc model."""
    def predict(self, df):
        return np.array([0.8] * len(df))


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors api/app.py feature logic for testing."""
    df = df.copy()
    df["ChargePerMin"] = df["MonthlyCharge"] / (df["DayMins"] + 1e-6)
    df["OverageRatio"] = df["OverageFee"] / (df["MonthlyCharge"] + 1e-6)
    df["RoamToDay"] = df["RoamMins"] / (df["DayMins"] + 1e-6)
    df["CallsPerDay"] = df["DayCalls"] / (df["AccountWeeks"] / 4 + 1e-6)
    df["HighServiceCalls"] = (df["CustServCalls"] >= 3).astype(int)
    return df


@pytest.fixture
def sample_input():
    return pd.DataFrame(
        [
            {
                "AccountWeeks": 128,
                "ContractRenewal": 1,
                "DataPlan": 1,
                "DataUsage": 2.7,
                "CustServCalls": 1,
                "DayMins": 265.1,
                "DayCalls": 110,
                "MonthlyCharge": 89.0,
                "OverageFee": 9.87,
                "RoamMins": 10.0,
            }
        ]
    )


def test_mock_predict_returns_score(sample_input):
    model = MockModel()
    df = apply_feature_engineering(sample_input)
    scores = model.predict(df)
    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0


def test_feature_engineering_columns(sample_input):
    df = apply_feature_engineering(sample_input)
    expected_new_cols = ["ChargePerMin", "OverageRatio", "RoamToDay", "CallsPerDay", "HighServiceCalls"]
    for col in expected_new_cols:
        assert col in df.columns, f"Missing engineered feature: {col}"


def test_high_service_calls_flag(sample_input):
    sample_input["CustServCalls"] = 4
    df = apply_feature_engineering(sample_input)
    assert df["HighServiceCalls"].iloc[0] == 1


def test_low_service_calls_flag(sample_input):
    sample_input["CustServCalls"] = 2
    df = apply_feature_engineering(sample_input)
    assert df["HighServiceCalls"].iloc[0] == 0


def test_prediction_threshold():
    model = MockModel()
    scores = model.predict(pd.DataFrame([{}]))
    prediction = int(scores[0] >= 0.5)
    assert prediction == 1  # score 0.8 should be churn
