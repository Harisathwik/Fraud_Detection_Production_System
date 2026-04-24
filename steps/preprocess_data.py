import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
from sklearn.compose import ColumnTransformer
from core.preprocessing import create_preprocessing_pipeline

@step
def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[
    Annotated[np.ndarray, "X_train_transformed"],
    Annotated[np.ndarray, "X_test_transformed"],
    Annotated[ColumnTransformer, "preprocessor"],
]:
    """Fits preprocessor and transforms data."""
    
    # Drop IDs and timestamps that aren't behavioral features
    # Drop velocity_flag (0.97 corr) and distance_from_home_km (0.88 corr) (Data Leakage)
    X_train = X_train.drop(columns=["transaction_id", "transaction_datetime", "velocity_flag", "distance_from_home_km"])
    X_test = X_test.drop(columns=["transaction_id", "transaction_datetime", "velocity_flag", "distance_from_home_km"])
    
    numeric_features = [
        "amount_usd", "transactions_last_1h",
        "transactions_last_24h", "avg_transaction_amount_last_30d",
        "amount_vs_avg_ratio", "days_since_last_transaction",
        "customer_age_years", "account_age_days"
    ]
    
    categorical_features = ["merchant_category", "merchant_country"]
    
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    print(f"Preprocessing complete. Transformed shape: {X_train_transformed.shape}")
    
    return X_train_transformed, X_test_transformed, preprocessor
