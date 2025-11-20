import mlflow
from ultralytics import YOLO
import os

# --- Configuration ---
# We use the exact path from your successful log output
BEST_PT_PATH = r"C:\Visual_qc_project\runs\yolo_train_82ce013c98ff4b779c64ab9188f1b73e\weights\best.pt"
MODEL_NAME = "visual-qc-yolov8"
MLFLOW_TRACKING_URI = "http://localhost:5000"

def register_existing_model():
    """Loads the saved best.pt and registers it to MLflow without retraining."""
    
    if not os.path.exists(BEST_PT_PATH):
        print(f"FATAL ERROR: Could not find model at {BEST_PT_PATH}")
        return

    print(f"Loading model from: {BEST_PT_PATH}")
    # Loading a FRESH model instance removes the 'unpicklable' data loaders
    model = YOLO(BEST_PT_PATH)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Visual-QC-YOLOv8-MVTec")

    print("Starting registration run...")
    with mlflow.start_run(run_name="Manual_Registration"):
        # 1. Log the physical file
        mlflow.log_artifact(BEST_PT_PATH, artifact_path="model_artifact")
        
        # 2. Register the model to the registry
        # Now that the model object is 'clean', this will succeed
        mlflow.pytorch.log_model(
            pytorch_model=model, 
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
        print(f"SUCCESS: Model registered as '{MODEL_NAME}' version 1.")

if __name__ == "__main__":
    register_existing_model()