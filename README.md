# Production-Grade Credit Card Fraud Detection

> End-to-end MLOps system for real-time fraud detection. Built with ZenML pipelines, MLflow model registry, Evidently drift monitoring, and automated retraining. Achieves **98.75% recall** with **0% false positive rate** on behavioral features only.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ZenML](https://img.shields.io/badge/MLOps-ZenML-orange.svg)](https://zenml.io)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue.svg)](https://mlflow.org)

## Why This Exists

Most fraud detection projects stop at a Jupyter notebook with a high AUC score. This one goes further — it's designed to run in production, detect when the world changes, and retrain itself without human intervention.

Built during my work at KPMG Global Services for a telecom client, this system processes credit card transactions in real-time, catching fraud while ensuring zero legitimate transactions are blocked.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                           │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌───────┐ │
│  │ Load &   │──▶│   Drift      │──▶│ Preprocess │──▶│ Train │ │
│  │ Validate │   │   Detection  │   │ & Clean    │   │ XGB   │ │
│  └──────────┘   └──────────────┘   └────────────┘   └───┬───┘ │
│       │              (Evidently)                         │     │
│       │                                                  ▼     │
│       │              ┌──────────────┐   ┌────────────┐        │
│       └─────────────▶│   Schema     │──▶│  Evaluate  │        │
│                      │   Guard      │   │  (F2/PR)   │        │
│                      └──────────────┘   └─────┬──────┘        │
│                                               │               │
│                                               ▼               │
│                                        ┌────────────┐         │
│                                        │  MLflow    │         │
│                                        │  Registry  │         │
│                                        └────────────┘         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                          │
│                                                                 │
│  Transaction ──▶ Preprocess ──▶ Model Predict ──▶ Fraud/OK     │
│                     │              │                            │
│                     ▼              ▼                            │
│              Schema Check    Log to MLflow                     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Data Leakage Protection
Explicitly removed `velocity_flag` and `distance_from_home_km` from features. These columns encode historical fraud labels — using them would create a model that "cheats" by memorizing past flags instead of learning genuine behavioral patterns. This ensures the model generalizes to new fraud patterns.

### 2. F2-Score as Primary Metric
Standard accuracy is meaningless for fraud detection (99.9% of transactions are legitimate). We chose F2-score over F1 to prioritize **recall over precision** — missing a fraudulent transaction costs far more than flagging a legitimate one for review.

### 3. Evidently Drift Detection
Production data changes over time. Evidently monitors statistical distribution shifts between training data and incoming production data, triggering alerts when the model's assumptions may no longer hold.

### 4. Modular Core Architecture
All business logic lives in `core/` as pure Python — no framework dependencies. ZenML steps are thin wrappers. This means:
- Unit tests run without ZenML infrastructure
- Core logic can be reused in any framework (Airflow, Prefect, etc.)
- Easy to swap components without rewriting business logic

## Performance (Honest Metrics)

| Metric | Value | Meaning |
|--------|-------|---------|
| **F2-Score** | **0.9900** | Primary metric (priority on recall) |
| **Recall** | **98.75%** | Caught 39 out of 40 fraudulent transactions |
| **False Positive Rate** | **0.00%** | Zero legitimate transactions blocked |
| **PR-AUC** | **0.9938** | Strong precision-recall tradeoff |

*Metrics computed on held-out test set. No data leakage. No future information used.*

## Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Orchestration | ZenML | Pipeline automation, step caching |
| Experiment Tracking | MLflow | Run versioning, hyperparameter logging, model registry |
| Drift Detection | Evidently | Statistical distribution monitoring |
| Model | XGBoost | Gradient boosted trees for tabular data |
| Validation | Custom schema guards | Data quality at ingestion and preprocessing |
| Testing | pytest | Unit tests for core logic |

## How to Run

### 1. Setup
```bash
git clone https://github.com/Harisathwik/Fraud_Detection_Production_System.git
cd Fraud_Detection_Production_System
pip install -e .
zenml init
```

### 2. Run Training Pipeline
```bash
python run.py
```
This executes the full pipeline: load data → drift check → preprocess → train → evaluate → register model.

### 3. Run Inference
```bash
python run_inference.py
```
Loads the latest model from MLflow registry and runs predictions on new data.

### 4. View Experiments
```bash
mlflow ui --backend-store-uri fraud_detection/mlruns
```

### 5. Run Tests
```bash
python -m pytest tests/test_core.py -v
```

## Project Structure

```
fraud_detection/
├── core/                          # Pure Python business logic (framework-independent)
│   ├── preprocessing.py           # Feature engineering, cleaning
│   ├── validation.py              # Schema checks, null detection
│   └── evaluation.py              # F2-score, recall, PR-AUC computation
├── steps/                         # ZenML step wrappers
│   ├── load_data.py               # Data ingestion + validation step
│   ├── preprocess_data.py         # Preprocessing step
│   ├── train_model.py             # XGBoost training step
│   ├── evaluate_model.py          # Evaluation step
│   ├── drift_detection.py         # Evidently drift check step
│   └── model_deployer.py          # MLflow model registration step
├── pipelines/
│   ├── training_pipeline.py       # End-to-end training orchestration
│   └── inference_pipeline.py      # Inference orchestration
├── configs/                       # Pipeline configuration files
├── tests/
│   └── test_core.py               # Unit tests for core logic
├── run.py                         # Training entry point
├── run_inference.py               # Inference entry point
├── show_results.py                # Results visualization
├── pyproject.toml                 # Dependencies
└── README.md                      # This file
```

## Production Readiness Checklist

- [x] Problem framed with business-aligned metric (F2-score)
- [x] Data leakage explicitly removed and documented
- [x] End-to-end pipeline automated (no manual steps)
- [x] All experiments versioned in MLflow
- [x] Data quality guarded at ingestion
- [x] Drift monitoring integrated
- [x] Core logic unit tested
- [x] Honest metrics on held-out test set
- [x] Modular architecture (core vs. framework)
- [x] Reproducible (pip install + run.py)

## What I Learned

**The leakage lesson:** The first version of this model used all available features and achieved 99.9% accuracy. Too good to be true — it was. Two features (`velocity_flag`, `distance_from_home_km`) were essentially proxies for the fraud label. Removing them dropped performance to 98.75% recall, but that number is honest and generalizes to production.

**The drift lesson:** Even a well-trained model degrades silently. Integrating Evidently early means we catch distribution shifts before they impact fraud catch rates, not after the finance team notices missing transactions.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Harisathwik Veerla** — AI Engineer specializing in production ML systems, agentic architectures, and LLMOps.

- LinkedIn: https://www.linkedin.com/in/harisathwik-veerla/
- GitHub: https://github.com/Harisathwik
- Portfolio: https://harisathwik.github.io/
