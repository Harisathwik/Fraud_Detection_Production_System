import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
from sklearn.model_selection import train_test_split
from core.validation import validate_schema, check_missing_values, check_class_imbalance

@step
def load_data(
    file_path: str,
    target_col: str = "is_fraud"
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Loads and validates data, then returns stratified splits."""
    df = pd.read_csv(file_path)
    
    # Expected columns for this project
    expected_cols = [
        "transaction_id", "transaction_datetime", "hour_of_day", "day_of_week",
        "is_weekend", "amount_usd", "merchant_category", "merchant_country",
        "is_foreign_transaction", "distance_from_home_km", "card_present",
        "chip_used", "pin_used", "billing_address_match", "cvv_match",
        "transactions_last_1h", "transactions_last_24h", 
        "avg_transaction_amount_last_30d", "amount_vs_avg_ratio",
        "days_since_last_transaction", "customer_age_years", "account_age_days",
        "is_new_merchant", "velocity_flag", "is_fraud"
    ]
    
    # Run core validations
    validate_schema(df, expected_cols)
    check_missing_values(df)
    check_class_imbalance(df, target_col)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Data loaded successfully. Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test
