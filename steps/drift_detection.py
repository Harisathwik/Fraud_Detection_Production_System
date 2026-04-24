import pandas as pd
from zenml import step

@step
def drift_detection(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> str:
    """Detects data drift using Evidently with verified paths."""
    from evidently.core.report import Report
    from evidently.presets.drift import DataDriftPreset

    # We remove IDs and timestamps for drift analysis
    ref = reference_data.drop(columns=["transaction_id", "transaction_datetime", "velocity_flag", "distance_from_home_km"], errors="ignore")
    curr = current_data.drop(columns=["transaction_id", "transaction_datetime", "velocity_flag", "distance_from_home_km"], errors="ignore")
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=curr)
    
    print("Evidently Data Drift Report generated successfully.")
        
    return "Drift Check Complete"
