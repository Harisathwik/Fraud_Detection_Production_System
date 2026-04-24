import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def create_preprocessing_pipeline(
    numeric_features: list, 
    categorical_features: list
) -> ColumnTransformer:
    """Creates a scikit-learn ColumnTransformer for fraud data."""
    
    numeric_transformer = Pipeline(steps=[
        ("scaler", RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough" # Keep binary flags as they are
    )

    return preprocessor
