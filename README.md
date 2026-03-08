# Telecom Churn Prediction — MLOps Project

Predict customer churn for a telecom provider using an XGBoost classifier, fully orchestrated with **uv · Git · DVC · DagsHub · MLflow · Docker · FastAPI · GitHub Actions**.

---

## Dataset

`data/raw/telecom_churn.csv` — 3,333 customers, 10 features, binary target (`Churn`).

| Feature | Description |
|---|---|
| AccountWeeks | Weeks as a customer |
| ContractRenewal | Renewed contract (1/0) |
| DataPlan | Has data plan (1/0) |
| DataUsage | Monthly data usage (GB) |
| CustServCalls | Customer service calls |
| DayMins | Total daytime minutes |
| DayCalls | Total daytime calls |
| MonthlyCharge | Monthly bill ($) |
| OverageFee | Overage fees ($) |
| RoamMins | Roaming minutes |

---

## Quick Start

### 1 — Clone & environment

```bash
git clone <GITHUB_REPO_URL>
cd telecom_churn_mlops

uv venv
source .venv/bin/activate
uv sync --frozen
```

### 2 — Configure secrets

```bash
cp .env.example .env
# Fill in DAGSHUB_USERNAME, DAGSHUB_TOKEN, MLFLOW_TRACKING_URI
```

### 3 — Set up DVC remote (DagsHub)

```bash
dvc remote add origin https://dagshub.com/<USERNAME>/telecom_churn_mlops.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <USERNAME>
dvc remote modify origin --local password <DAGSHUB_TOKEN>
dvc remote default origin
```

### 4 — Track raw data

```bash
dvc add data/raw/telecom_churn.csv
git add data/raw/telecom_churn.csv.dvc data/raw/.gitignore
git commit -m "data: add raw telecom churn dataset"
dvc push
```

### 5 — Run the full pipeline

```bash
dvc repro          # smart cached — only reruns changed stages
dvc metrics show   # print test metrics
dvc push           # push artifacts to DagsHub
```

### 6 — View experiments

```bash
mlflow ui          # open http://localhost:5000
```

---

## Project Structure

```
telecom_churn_mlops/
├── data/
│   ├── raw/                  # original data (DVC-tracked)
│   ├── processed/            # train/test splits (DVC-tracked)
│   └── features/             # engineered features (DVC-tracked)
├── src/
│   ├── data_ingestion.py     # load + validate raw data
│   ├── preprocessing.py      # clean + train/test split
│   ├── feature_engineering.py# domain features + scaling
│   ├── train.py              # XGBoost + MLflow logging
│   ├── evaluate.py           # test metrics + plot CSVs
│   └── predict.py            # batch inference CLI
├── api/
│   └── app.py                # FastAPI serving endpoint
├── models/                   # saved artifacts (DVC-tracked)
├── metrics/                  # JSON metrics + plot CSVs
├── tests/
│   ├── test_preprocessing.py
│   └── test_predict.py
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── notebooks/
│   └── 01_eda.ipynb
├── .github/workflows/ci.yml  # CI/CD pipeline
├── dvc.yaml                  # pipeline definition
├── params.yaml               # all hyperparameters
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## DVC Pipeline

```
preprocess → feature_engineering → train → evaluate
```

```bash
dvc dag            # visualise DAG
dvc params diff    # show param changes since last run
dvc repro --force  # force re-run all stages
```

---

## MLflow Model Registry

```python
import mlflow

# Register best run
mlflow.register_model(f"runs:/{run_id}/model", "TelecomChurnModel")

# Promote to Production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="TelecomChurnModel", version=1, stage="Production"
)
```

---

## Serving

### Local (development)

```bash
uvicorn api.app:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the auto-generated Swagger UI.

### Docker

```bash
docker compose up --build serve
```

### Example API call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AccountWeeks": 65,
    "ContractRenewal": 0,
    "DataPlan": 0,
    "DataUsage": 0.0,
    "CustServCalls": 4,
    "DayMins": 300.5,
    "DayCalls": 95,
    "MonthlyCharge": 75.0,
    "OverageFee": 12.5,
    "RoamMins": 5.0
  }'
```

Response:
```json
{
  "churn_score": 0.874,
  "churn_prediction": 1,
  "risk_level": "High"
}
```

---

## CI/CD (GitHub Actions)

Required secrets in **Settings → Secrets → Actions**:

| Secret | Value |
|---|---|
| `DAGSHUB_TOKEN` | DagsHub personal access token |
| `DAGSHUB_USERNAME` | Your DagsHub username |
| `MLFLOW_TRACKING_URI` | Full DagsHub MLflow URI |

On every push to `main`: runs tests → pulls data → reruns pipeline → pushes artifacts → logs MLflow run.

---

## Daily Workflow Cheatsheet

| Start of day | During development | End of day |
|---|---|---|
| `git pull --rebase` | `dvc repro` | `git add -p` |
| `dvc pull` | `dvc metrics show` | `git commit -m "..."` |
| `source .venv/bin/activate` | `mlflow ui` | `dvc push && git push` |

---

## Engineered Features

In addition to the 10 raw features, the pipeline adds:

| Feature | Formula | Rationale |
|---|---|---|
| `ChargePerMin` | MonthlyCharge / DayMins | Revenue efficiency |
| `OverageRatio` | OverageFee / MonthlyCharge | Overage burden |
| `RoamToDay` | RoamMins / DayMins | Roaming usage share |
| `CallsPerDay` | DayCalls / (AccountWeeks/4) | Call frequency |
| `HighServiceCalls` | CustServCalls ≥ 3 | Known churn indicator |
