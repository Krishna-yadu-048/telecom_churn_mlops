# Telecom Churn Prediction — End-to-End MLOps Pipeline

[![CI](https://github.com/Krishna-yadu-048/telecom_churn_mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Krishna-yadu-048/telecom_churn_mlops/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Model](https://img.shields.io/badge/model-XGBoost-orange)
![Tracking](https://img.shields.io/badge/tracking-MLflow%20%7C%20DagsHub-0194E2)
![Serving](https://img.shields.io/badge/serving-FastAPI-009688)
![License](https://img.shields.io/badge/license-MIT-green)

A production-style MLOps project that predicts telecom customer churn with an **XGBoost** classifier. It covers the full lifecycle — reproducible data versioning, a cached training pipeline, experiment tracking, a model registry, containerized serving, and CI/CD — using **uv · DVC · DagsHub · MLflow · Docker · FastAPI · GitHub Actions**.

The goal of this repo is to demonstrate the gap-closing work between an ML notebook and a deployed, reproducible service: every artifact is versioned, every run is tracked, and every push is tested and retrained automatically.

---

## Highlights

- **Reproducible pipeline** — a 4-stage DVC DAG (`preprocess → feature_engineering → train → evaluate`) with smart caching, so only changed stages rerun.
- **Data & artifact versioning** — raw data, splits, features, and the trained model are tracked with DVC and stored on a DagsHub remote.
- **Experiment tracking & registry** — every training run is logged to MLflow (hosted on DagsHub); the best model is registered and promoted to `Production`.
- **Class-imbalance handling** — `imbalanced-learn` is available for resampling the skewed churn target.
- **Containerized** — separate `Dockerfile.train` and `Dockerfile.serve` images, orchestrated with `docker-compose`.
- **Real-time serving** — a FastAPI app loads the `Production` model from the MLflow registry and exposes `/predict` and `/health`, with auto-generated Swagger docs.
- **CI/CD** — GitHub Actions runs tests on every push, then pulls data, reruns the pipeline, and pushes artifacts on `main`.

---

## Model Performance

Current test-set metrics (from `metrics/test_metrics.json`):

| Metric | Score |
|---|---|
| ROC AUC | 0.855 |
| F1 | 0.721 |
| Precision | 0.827 |
| Recall | 0.639 |

> Optimized for ROC AUC (the configured `primary_metric`). Precision is prioritized over recall, reflecting a cost preference for confident churn flags over catching every churner.

---

## Architecture

```
                         ┌──────────────┐
   data/raw  ──DVC──►    │  preprocess  │  ──► train.csv / test.csv
                         └──────┬───────┘
                                ▼
                         ┌──────────────────────┐
                         │ feature_engineering  │  ──► *_features.csv
                         └──────┬───────────────┘
                                ▼
                         ┌──────────────┐         ┌──────────────────┐
                         │    train     │  ─────► │ MLflow (DagsHub) │
                         │  (XGBoost)   │         │  runs + registry │
                         └──────┬───────┘         └────────┬─────────┘
                                ▼                          │ Production model
                         ┌──────────────┐                  ▼
                         │   evaluate   │           ┌──────────────────┐
                         │ metrics+plots│           │ FastAPI /predict │
                         └──────────────┘           └──────────────────┘
```

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

### Engineered features

The feature stage adds five domain-driven features on top of the 10 raw ones:

| Feature | Formula | Rationale |
|---|---|---|
| `ChargePerMin` | MonthlyCharge / DayMins | Revenue efficiency |
| `OverageRatio` | OverageFee / MonthlyCharge | Overage burden |
| `RoamToDay` | RoamMins / DayMins | Roaming usage share |
| `CallsPerDay` | DayCalls / (AccountWeeks / 4) | Call frequency |
| `HighServiceCalls` | CustServCalls ≥ 3 | Known churn indicator |

---

## Tech Stack

| Concern | Tool |
|---|---|
| Dependency management | uv |
| Data & artifact versioning | DVC |
| Remote storage & MLflow host | DagsHub |
| Experiment tracking & registry | MLflow 2.19 |
| Model | XGBoost (`imbalanced-learn` for resampling) |
| Serving | FastAPI + Uvicorn |
| Containers | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Testing / linting | pytest · ruff · black |

---

## Quick Start

### 1 — Clone & environment

```bash
git clone https://github.com/Krishna-yadu-048/telecom_churn_mlops.git
cd telecom_churn_mlops

uv venv
source .venv/bin/activate
uv sync --extra dev
```

### 2 — Configure secrets

```bash
cp .env.example .env
# Fill in DAGSHUB_USERNAME, DAGSHUB_TOKEN, MLFLOW_TRACKING_URI
```

### 3 — Set up the DVC remote (DagsHub)

```bash
dvc remote add origin https://dagshub.com/<USERNAME>/telecom_churn_mlops.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <USERNAME>
dvc remote modify origin --local password <DAGSHUB_TOKEN>
dvc remote default origin
```

### 4 — Pull data & run the pipeline

```bash
dvc pull           # fetch DVC-tracked data/artifacts
dvc repro          # smart cached — only reruns changed stages
dvc metrics show   # print test metrics
dvc push           # push artifacts to DagsHub
```

### 5 — View experiments

```bash
mlflow ui          # local UI at http://localhost:5000
```

(MLflow runs are also viewable on DagsHub via your `MLFLOW_TRACKING_URI`.)

---

## DVC Pipeline

```
preprocess → feature_engineering → train → evaluate
```

| Stage | Script | Outputs |
|---|---|---|
| preprocess | `src/preprocessing.py` | `data/processed/{train,test}.csv` |
| feature_engineering | `src/feature_engineering.py` | `data/features/{train,test}_features.csv` |
| train | `src/train.py` | `models/TelecomChurnModel.pkl`, `metrics/train_metrics.json` |
| evaluate | `src/evaluate.py` | `metrics/test_metrics.json`, ROC & confusion-matrix plots |

```bash
dvc dag            # visualise the DAG
dvc params diff    # show param changes since last run
dvc repro --force  # force re-run of all stages
```

All hyperparameters live in `params.yaml` (data split, scaling, XGBoost settings, eval threshold, MLflow names) — edit there, then `dvc repro` picks up the change.

---

## MLflow Model Registry

```python
import mlflow

# Register the best run
mlflow.register_model(f"runs:/{run_id}/model", "TelecomChurnModel")

# Promote to Production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="TelecomChurnModel", version=1, stage="Production"
)
```

The serving app loads `models:/TelecomChurnModel/Production`, so promoting a new version is enough to roll it out.

---

## Serving

### Local (development)

```bash
uvicorn api.app:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the auto-generated Swagger UI.

### Docker

```bash
docker compose up --build serve   # serving API on :8000
docker compose up --build train   # run the training pipeline in a container
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Score a single customer |
| GET | `/health` | Liveness + whether the model loaded |
| GET | `/docs` | Swagger UI |

### Example request

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

```json
{
  "churn_score": 0.874,
  "churn_prediction": 1,
  "risk_level": "High"
}
```

`risk_level` is bucketed from the score: `High ≥ 0.7`, `Medium ≥ 0.4`, else `Low`. The API recomputes the five engineered features internally, so callers only send the 10 raw inputs.

### Batch / CLI inference

```bash
python src/predict.py   # batch inference using the registered model
```

---

## CI/CD (GitHub Actions)

Workflow: `.github/workflows/ci.yml`

- **`test`** — runs on every push: `uv sync` then `pytest tests/ -v`.
- **`train`** — runs on `main` after tests pass: configures the DVC remote, `dvc pull` → `dvc repro` → `dvc push`, then logs the training run to MLflow.

Required repository secrets (**Settings → Secrets and variables → Actions**):

| Secret | Value |
|---|---|
| `DAGSHUB_USERNAME` | Your DagsHub username |
| `DAGSHUB_TOKEN` | DagsHub personal access token |
| `MLFLOW_TRACKING_URI` | Full DagsHub MLflow URI |

---

## Testing

```bash
uv run pytest tests/ -v
```

Covers preprocessing logic (`test_preprocessing.py`) and prediction/feature construction (`test_predict.py`).

---

## Project Structure

```
telecom_churn_mlops/
├── data/
│   ├── raw/                   # original data (DVC-tracked)
│   ├── processed/             # train/test splits (DVC-tracked)
│   └── features/              # engineered features (DVC-tracked)
├── src/
│   ├── data_ingestion.py      # load + validate raw data
│   ├── preprocessing.py       # clean + train/test split
│   ├── feature_engineering.py # domain features + scaling
│   ├── train.py               # XGBoost + MLflow logging
│   ├── evaluate.py            # test metrics + plot CSVs
│   └── predict.py             # batch inference CLI
├── api/
│   └── app.py                 # FastAPI serving endpoint
├── tests/
│   ├── test_preprocessing.py
│   └── test_predict.py
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── notebooks/
│   └── 01_eda.ipynb           # exploratory data analysis
├── metrics/                   # JSON metrics + plot CSVs
├── models/                    # saved artifacts (DVC-tracked)
├── .github/workflows/ci.yml   # CI/CD pipeline
├── dvc.yaml                   # pipeline definition
├── params.yaml                # all hyperparameters
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
