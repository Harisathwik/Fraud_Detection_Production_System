import pandas as pd
import numpy as np
from zenml import step
import mlflow
from sklearn.base import ClassifierMixin
from core.evaluation import calculate_fraud_metrics

@step(experiment_tracker="mlflow_tracker")
def evaluate_model(
    model: ClassifierMixin,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> dict:
    """Evaluates the model and logs metrics to MLflow."""
    
    y_pred = model.predict(X_test)
    metrics = calculate_fraud_metrics(y_test, y_pred)
    
    # Log metrics to MLflow explicitly for comparison
    mlflow.log_metrics(metrics)
    
    print(f"Evaluation Complete:")
    print(f"F2-Score: {metrics['f2_score']:.4f}")
    print(f"Recall (Fraud Catch Rate): {metrics['recall']:.4f}")
    print(f"False Positive Rate (Friction): {metrics['fpr']:.4f}")
    
    return metrics
