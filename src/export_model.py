import mlflow
import os
from ultralytics import YOLO

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "visual-qc-yolov8"
EXPORT_FORMAT = "onnx"

def export_optimized_model():
    """
    Fetches the latest PyTorch model from the MLflow Registry, 
    exports it to the ONNX format, and logs the ONNX model back.
    """
    
    # 1. Setup MLflow Connection
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    
    print("Connecting to MLflow and fetching the latest registered model version...")
    
    # 2. Fetch the Latest Registered Model Version
    try:
        all_versions = client.get_latest_versions(MODEL_NAME, stages=None)
        # Sort by version number (descending) and get the newest one
        latest_version = sorted(all_versions, key=lambda x: int(x.version), reverse=True)[0]
        
        run_id = latest_version.run_id
        print(f"Found Model Version: {latest_version.version} associated with Run ID: {run_id}")

    except Exception as e:
        print(f"Error fetching model from MLflow: {e}")
        return

    # 3. Download the .pt model artifact using the Run ID
    # We fetch the file 'best.pt' from the 'model_artifact' folder we created in register_manual.py
    print("Downloading model artifact...")
    try:
        local_pt_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model_artifact/best.pt",
            dst_path="./artifacts"
        )
    except Exception as e:
        print(f"Could not find 'model_artifact/best.pt'. Trying fallback locations...")
        # Fallback: sometimes it might be just 'best.pt' or inside 'weights' depending on the run
        local_pt_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="weights/best.pt", 
            dst_path="./artifacts"
        )

    print(f"Loaded PyTorch model artifact from: {local_pt_path}")
    
    # 4. Load the PyTorch model and Export to ONNX
    model = YOLO(local_pt_path)
    
    print(f"Exporting model to {EXPORT_FORMAT}...")
    # Export to ONNX
    export_path = model.export(
        format=EXPORT_FORMAT, 
        simplify=True,           
        dynamic=True             
    )
    
    print(f"Model exported successfully to ONNX at: {export_path}")
    
    # 5. Log the ONNX model back to MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(export_path, artifact_path="onnx_model")
        mlflow.log_param("export_format", EXPORT_FORMAT)
        print("ONNX model logged back to MLflow.")

if __name__ == "__main__":
    export_optimized_model()