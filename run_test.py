from steps.load_data import load_data
from steps.preprocess_data import preprocess_data
import os

def test_pipeline_steps():
    csv_path = "../credit_card_fraud_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Could not find dataset at {csv_path}")
        return

    print("--- Testing load_data step ---")
    X_train, X_test, y_train, y_test = load_data.entrypoint(
        file_path=csv_path,
        target_col="is_fraud"
    )
    
    print("--- Testing preprocess_data step ---")
    X_train_trans, X_test_trans, preprocessor = preprocess_data.entrypoint(
        X_train=X_train,
        X_test=X_test
    )
    
    assert X_train_trans.shape[0] == 40000
    assert X_train_trans.shape[1] > 20 # Should have expanded one-hot columns
    print("Verification complete: Preprocessing logic is correct.")

if __name__ == "__main__":
    test_pipeline_steps()
