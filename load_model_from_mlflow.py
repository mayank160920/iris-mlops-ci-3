# Load and save the latest model from MLflow Model Registry
import os
import mlflow 
import joblib
from mlflow.tracking import MlflowClient

def load_model_from_mlflow(registered_model_name: str, model_save_path: str, tracking_uri: str) -> None:
    """Load the latest version of a registered model from MLflow Model Registry and save it locally.

    Args:
        registered_model_name (str): Name of the registered model in MLflow.
        model_save_path (str): Local path to save the loaded model.
    """
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow Tracking URI set to: {tracking_uri}")
    client = MlflowClient()
    latest_model = client.get_latest_versions(registered_model_name)[0]
    model_uri = f"models:/{registered_model_name}/{latest_model.version}"
    print(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    joblib.dump(model, model_save_path)
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "IrisDecisionTreeModel")
    MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "model.joblib")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    load_model_from_mlflow(REGISTERED_MODEL_NAME, MODEL_SAVE_PATH, MLFLOW_TRACKING_URI)