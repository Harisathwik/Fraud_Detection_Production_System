import pandas as pd
import numpy as np
from typing import List, Dict

def validate_schema(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """Checks if the dataframe has all expected columns."""
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    return True

def check_missing_values(df: pd.DataFrame, threshold: float = 0.05):
    """Flags if any column has more than 'threshold' percent missing values."""
    null_counts = df.isnull().mean()
    high_null_cols = null_counts[null_counts > threshold]
    if not high_null_cols.empty:
        raise ValueError(f"High missing values detected in: {high_null_cols.to_dict()}")

def check_class_imbalance(df: pd.DataFrame, target_col: str):
    """Calculates and logs the class distribution."""
    dist = df[target_col].value_counts(normalize=True).to_dict()
    print(f"Target distribution: {dist}")
    if dist.get(1, 0) < 0.001:
         print("WARNING: Extremely low positive class count detected (<0.1%)")
    return dist
