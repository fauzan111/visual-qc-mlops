import os
import mlflow
import torch
from ultralytics import YOLO

# --- Configuration ---
# 1. MLflow Tracking URI (Should match your running MLflow UI instance)
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Visual-QC-YOLOv8-MVTec"
MODEL_NAME = "visual-qc-yolov8"

# 2. Data and Model Configuration
DATA_YAML_PATH = "data.yaml"
# 'n' for nano (fastest, best for edge); use 's' or 'm' for higher accuracy
MODEL_SIZE = "yolov8n.pt" 

# 3. Training Hyperparameters
EPOCHS = 50 
IMG_SIZE = 640
BATCH_SIZE = 16 
# Note: Lower batch size if GPU memory is limited, or for edge simulation

def train_and_log_model():
    """Trains the YOLOv8 model and logs results and model to MLflow."""
    
    # 1. Setup MLflow Connection and Experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Set the experiment name. If it doesn't exist, MLflow creates it.
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # 2. Log Parameters to MLflow
        mlflow.log_params({
            "epochs": EPOCHS,
            "imgsz": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "model_size": MODEL_SIZE,
            "data_yaml": DATA_YAML_PATH
        })

        # 3. Initialize and Train Model
        print(f"Starting training for model: {MODEL_SIZE}...")
        model = YOLO(MODEL_SIZE) # Load pre-trained weights
        
        # Start training
        results = model.train(
            data=DATA_YAML_PATH,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            # Name the run based on the MLflow ID to easily find artifacts later
            project="runs",
            name=f"yolo_train_{run_id}",
            exist_ok=True,
            verbose=False 
        )

        print("Training complete. Logging final validation metrics...")

        # 4. Log Metrics
        # YOLOv8 stores final validation metrics in the results object
        metrics = results.metrics 
        mlflow.log_metrics({
            "val/mAP50": metrics.get('metrics/mAP50(B)', 0.0),
            "val/mAP50-95": metrics.get('metrics/mAP50-95(B)', 0.0),
            "val/precision": metrics.get('metrics/precision(B)', 0.0),
            "val/recall": metrics.get('metrics/recall(B)', 0.0)
        })

        # 5. Model Registration (Artifact & Registry)
        
        # Determine the path where YOLOv8 saves the final model artifact
        yolo_save_dir = os.path.join("runs", f"yolo_train_{run_id}", "weights")
        best_model_path = os.path.join(yolo_save_dir, "best.pt")
        
        if os.path.exists(best_model_path):
            
            # Log the model artifact directly to MLflow (for raw file access)
            mlflow.log_artifact(best_model_path, artifact_path="model_artifact")
            
            # Use mlflow.pytorch.log_model for proper model tracking and Registry integration
            mlflow.pytorch.log_model(
                pytorch_model=model, # Use the model object trained by YOLO
                artifact_path="model",
                registered_model_name=MODEL_NAME, # This registers the model for export_model.py
                # This tag indicates the stage for the first log
                tags={"stage": "Staging"} 
            )
            print(f"Model successfully logged and registered as '{MODEL_NAME}' in the MLflow Model Registry.")
        else:
            print(f"ERROR: Best model not found at {best_model_path}. Training might have failed.")


if __name__ == "__main__":
    if not os.path.exists(DATA_YAML_PATH):
        print(f"FATAL ERROR: The required data config file '{DATA_YAML_PATH}' was not found.")
        print("Please ensure you have created this file in your project root.")
    else:
        # Check for CUDA availability (optional, but good practice for deep learning)
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU for training. This may take a long time.")
            
        train_and_log_model()