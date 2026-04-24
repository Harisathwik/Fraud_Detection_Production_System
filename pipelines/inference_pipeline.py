from zenml import pipeline, step
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.client import Client

@pipeline
def fraud_inference_pipeline():
    """Pipeline to deploy the latest model from the registry."""
    
    # We fetch the model artifact from the last training run correctly
    client = Client()
    training_run = client.get_pipeline("fraud_training_pipeline").last_run
    
    # Get the specific step using the .steps dictionary/property
    model_artifact = training_run.steps["train_model"].output
    
    # Deploy using the artifact
    mlflow_model_deployer_step(
        model=model_artifact,
        model_name="fraud_detection_model",
        workers=1,
        timeout=300
    )
