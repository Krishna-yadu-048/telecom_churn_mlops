"""
FastAPI serving app for Telecom Churn Prediction.
Loads the Production model from MLflow registry on startup.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Predicts the probability of customer churn using an XGBoost model.",
    version="1.0.0",
)

# Load model at startup from MLflow registry
MODEL_NAME = os.getenv("MODEL_NAME", "TelecomChurnModel")
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model loaded from {model_uri}")
    except Exception as e:
        logger.error(f"Could not load model: {e}")


class CustomerFeatures(BaseModel):
    AccountWeeks: int = Field(..., example=128, description="Number of weeks as a customer")
    ContractRenewal: int = Field(..., example=1, description="1 if renewed contract, else 0")
    DataPlan: int = Field(..., example=1, description="1 if has data plan, else 0")
    DataUsage: float = Field(..., example=2.7, description="Monthly data usage (GB)")
    CustServCalls: int = Field(..., example=1, description="Number of customer service calls")
    DayMins: float = Field(..., example=265.1, description="Total daytime minutes used")
    DayCalls: int = Field(..., example=110, description="Total daytime calls made")
    MonthlyCharge: float = Field(..., example=89.0, description="Monthly bill amount ($)")
    OverageFee: float = Field(..., example=9.87, description="Overage fees charged ($)")
    RoamMins: float = Field(..., example=10.0, description="Total roaming minutes used")


class PredictionResponse(BaseModel):
    churn_score: float
    churn_prediction: int
    risk_level: str


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    df = pd.DataFrame([data.dict()])

    # Add engineered features (must match feature_engineering.py)
    df["ChargePerMin"] = df["MonthlyCharge"] / (df["DayMins"] + 1e-6)
    df["OverageRatio"] = df["OverageFee"] / (df["MonthlyCharge"] + 1e-6)
    df["RoamToDay"] = df["RoamMins"] / (df["DayMins"] + 1e-6)
    df["CallsPerDay"] = df["DayCalls"] / (df["AccountWeeks"] / 4 + 1e-6)
    df["HighServiceCalls"] = (df["CustServCalls"] >= 3).astype(int)

    score = float(model.predict(df)[0])
    prediction = int(score >= 0.5)

    if score >= 0.7:
        risk = "High"
    elif score >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return PredictionResponse(
        churn_score=round(score, 4),
        churn_prediction=prediction,
        risk_level=risk,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
