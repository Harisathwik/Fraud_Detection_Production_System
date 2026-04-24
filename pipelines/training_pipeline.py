from zenml import pipeline
from steps.load_data import load_data
from steps.preprocess_data import preprocess_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from steps.drift_detection import drift_detection

@pipeline
def fraud_training_pipeline(
    data_path: str,
    target_col: str = "is_fraud"
):
    """Production training pipeline with honest metrics and tracking."""
    
    # 1. Load and Split
    X_train, X_test, y_train, y_test = load_data(file_path=data_path, target_col=target_col)
    
    # 2. Drift Detection
    drift_status = drift_detection(reference_data=X_train, current_data=X_test)
    
    # 3. Preprocess (Leaked features removed: velocity_flag, distance_from_home_km)
    X_train_trans, X_test_trans, preprocessor = preprocess_data(X_train=X_train, X_test=X_test)
    
    # 4. Train production model (XGBoost)
    model = train_model(X_train=X_train_trans, y_train=y_train)
    
    # 5. Evaluate
    metrics = evaluate_model(model=model, X_test=X_test_trans, y_test=y_test)
    
    return model, preprocessor, metrics
