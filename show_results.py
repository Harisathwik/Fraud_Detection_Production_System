from zenml.client import Client
import pandas as pd

def show_final_results():
    client = Client()
    try:
        pipeline = client.get_pipeline("fraud_training_pipeline")
        last_run = pipeline.last_run
        
        print(f"--- Pipeline: {pipeline.name} ---")
        print(f"Run Name: {last_run.name}")
        print(f"Status: {last_run.status}")
        
        # Correctly load artifact in newer ZenML versions
        eval_step = last_run.steps["evaluate_model"]
        metrics = eval_step.output.load()
        
        print("\n--- Final Production Metrics ---")
        print(f"F2-Score: {metrics['f2_score']:.4f} (Priority Metric)")
        print(f"Recall: {metrics['recall']:.4f} (Fraud Catch Rate)")
        print(f"FPR: {metrics['fpr']:.4f} (Customer Friction)")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        
        print("\nEvery hyperparameter and artifact is versioned in ZenML/MLflow.")
        
    except Exception as e:
        print(f"Error fetching results: {e}")

if __name__ == "__main__":
    show_final_results()
