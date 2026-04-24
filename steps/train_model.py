import pandas as pd
import numpy as np
from zenml import step
import mlflow
from xgboost import XGBClassifier
from sklearn.base import ClassifierMixin

@step(experiment_tracker="mlflow_tracker")
def train_model(
    X_train: np.ndarray,
    y_train: pd.Series,
) -> ClassifierMixin:
    """Trains a production-grade XGBoost model and logs it correctly."""
    
    # scale_pos_weight = total_negative_examples / total_positive_examples
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=124, 
        eval_metric="aucpr", 
        random_state=42
    )
    
    print("Training production XGBoost model...")
    model.fit(X_train, y_train)
    
    # Explicitly log the model to MLflow with the expected name for the deployer
    mlflow.sklearn.log_model(model, "model")
    
    return model
