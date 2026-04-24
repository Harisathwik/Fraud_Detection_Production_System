from pipelines.inference_pipeline import fraud_inference_pipeline
from zenml.client import Client

def main():
    # Programmatically enforce the correct stack
    client = Client()
    try:
        client.activate_stack("fraud_stack")
        print("Active stack enforced: fraud_stack")
    except Exception as e:
        print(f"Warning: Could not activate fraud_stack: {e}")

    print("--- Starting Inference (Deployment) Pipeline ---")
    fraud_inference_pipeline()
    print("--- Deployment Complete ---")
    
    # Show the deployment status
    model_deployer = client.active_stack.model_deployer
    services = model_deployer.find_model_server(
        pipeline_name="fraud_inference_pipeline",
        running=True
    )
    
    if services:
        print(f"Model is LIVE at: {services[0].prediction_url}")
    else:
        print("No active model server found. Check ZenML dashboard for errors.")

if __name__ == "__main__":
    main()
