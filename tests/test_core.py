import pytest
import pandas as pd
import numpy as np
from core.validation import validate_schema, check_missing_values
from core.preprocessing import create_preprocessing_pipeline

def test_validate_schema_success():
    df = pd.DataFrame({"col1": [1], "col2": [2]})
    assert validate_schema(df, ["col1", "col2"]) is True

def test_validate_schema_failure():
    df = pd.DataFrame({"col1": [1]})
    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(df, ["col1", "col2"])

def test_check_missing_values():
    df = pd.DataFrame({"col1": [1, None, 3], "col2": [1, 2, 3]})
    # 33% missing in col1 > 5% threshold
    with pytest.raises(ValueError, match="High missing values"):
        check_missing_values(df, threshold=0.05)

def test_preprocessing_pipeline_creation():
    numeric_features = ["num1"]
    categorical_features = ["cat1"]
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    
    # Verify preprocessor is correctly built
    assert preprocessor.transformers[0][0] == "num"
    assert preprocessor.transformers[1][0] == "cat"

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__])
