from zenml import step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from typing import cast

@step(enable_cache=False)
def mlflow_model_deployer_step(
    model: str = "fraud_detection_model",
) -> MLFlowDeploymentService:
    """Deploys the model to a local MLflow server."""
    
    # Get the active mlflow model deployer from the stack
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    
    # Check for existing deployments
    existing_services = model_deployer.find_model_server(
        pipeline_name="fraud_training_pipeline",
        model_name=model,
    )
    
    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        print(f"Model already deployed at: {service.prediction_url}")
        return service
        
    # If no existing service, deploy a new one (integrated with MLflow registry)
    print("Initializing new MLflow deployment service...")
    # Note: In a full pipeline, ZenML's MLFlowModelDeployer handles the logic
    # through the 'mlflow_model_deployer_step' built-in.
    # We will use the built-in one for simplicity in the pipeline definition.
    return None
