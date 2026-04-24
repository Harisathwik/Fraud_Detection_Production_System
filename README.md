# Production-Grade Credit Card Fraud Detection

An automated, honest, and reproducible MLOps system for identifying fraudulent transactions in real-time.

## 🚀 Overview
This system uses **XGBoost** and **ZenML** to build a high-recall fraud detection model. We prioritized the **F2-Score** to catch 98.75% of fraud while maintaining zero legitimate customer friction.

### Key MLOps Features
- **Data Leakage Protection**: Explicitly removed `velocity_flag` and `distance_from_home_km` to ensure the model learns behavioral patterns, not historical cheating.
- **Drift Detection**: Integrated **Evidently** to monitor statistical distribution shifts between training and production data.
- **Reproducibility**: Every run, artifact, and hyperparameter is versioned in **MLflow**.
- **Modular Core**: Core logic is isolated in `core/` for framework-independent testing.

## 📊 Final Performance (Honest Metrics)
| Metric | Value | Meaning |
|--------|-------|---------|
| **F2-Score** | **0.9900** | Primary metric (Priority on Recall) |
| **Recall** | **98.75%** | Successfully caught nearly all fraud |
| **False Positive Rate** | **0.00%** | Minimal friction for legitimate users |
| **PR-AUC** | **0.9938** | Strong precision-recall performance |

## 🛠️ How to Run

### 1. Setup Environment
```bash
pip install -e .
zenml init
zenml stack set fraud_stack
```

### 2. Run Training Pipeline
```bash
python run.py
```

### 3. View Results (MLflow)
```bash
mlflow ui --backend-store-uri fraud_detection/mlruns
```

### 4. Run Unit Tests
```bash
python -m pytest tests/test_core.py
```

## 🏗️ Project Structure
- `core/`: Pure Python logic (Preprocessing, Validation, Evaluation).
- `steps/`: ZenML wrappers for the core logic.
- `pipelines/`: Automated training and drift detection workflows.
- `tests/`: Automated unit tests.
- `run.py`: Main entry point to trigger retraining.
