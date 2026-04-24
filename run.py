from pipelines.training_pipeline import fraud_training_pipeline
import os
from zenml.client import Client

def main():
    data_path = "../credit_card_fraud_dataset.csv"

    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return

    # Programmatically enforce the correct stack
    client = Client()
    try:
        client.activate_stack("fraud_stack")
        print("Active stack enforced: fraud_stack")
    except Exception as e:
        print(f"Warning: Could not activate fraud_stack: {e}")

    print("--- Starting Production Training Pipeline ---")
    fraud_training_pipeline(data_path=data_path)
    print("--- Pipeline Execution Finished ---")
    
    print("\nModel trained and evaluated successfully.")
    print("Every experiment, metric, and artifact is tracked in MLflow.")
    print("To view results, run: mlflow ui --backend-store-uri fraud_detection/mlruns")

if __name__ == "__main__":
    main()
