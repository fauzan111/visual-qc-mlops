import os
import mlflow
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

def train_and_log_model():
    """Trains the YOLOv8 model and logs results and model to MLflow."""
    
    # 1. Setup MLflow Connection and Experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Starting MLflow Run in experiment: {EXPERIMENT_NAME}")

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
        print(f"Loading model: {MODEL_SIZE}...")
        model = YOLO(MODEL_SIZE) 
        
        print("Starting training...")
        # Start training
        results = model.train(
            data=DATA_YAML_PATH,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project="runs",
            name=f"yolo_train_{run_id}",
            exist_ok=True,
            verbose=True
        )

        print("Training complete. Logging metrics...")

        # 4. Log Metrics
        # YOLOv8 stores final validation metrics in the results object
        metrics = results.metrics if hasattr(results, 'metrics') else results.results_dict
        
        # Depending on YOLO version, metrics might be accessed differently. 
        # We try to fetch standard keys safely.
        if metrics:
            mlflow.log_metrics({
                "val/mAP50": metrics.get('metrics/mAP50(B)', 0.0),
                "val/mAP50-95": metrics.get('metrics/mAP50-95(B)', 0.0),
                "val/precision": metrics.get('metrics/precision(B)', 0.0),
                "val/recall": metrics.get('metrics/recall(B)', 0.0)
            })

        # 5. Model Registration
        # Determine the path where YOLOv8 saves the final model artifact
        yolo_save_dir = os.path.join("runs", f"yolo_train_{run_id}", "weights")
        best_model_path = os.path.join(yolo_save_dir, "best.pt")
        
        if os.path.exists(best_model_path):
            print(f"Found best model at: {best_model_path}")
            
            # Log the model artifact directly
            mlflow.log_artifact(best_model_path, artifact_path="model_artifact")
            
            # Register the model for versioning
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME
            )
            print(f"Model registered as '{MODEL_NAME}' in MLflow.")
        else:
            print(f"WARNING: Could not find model file at {best_model_path}")

if __name__ == "__main__":
    # Ensure data.yaml exists
    if not os.path.exists(DATA_YAML_PATH):
        print(f"FATAL ERROR: {DATA_YAML_PATH} not found.")
        print("Please ensure you created the data.yaml file in the project root.")
    else:
        train_and_log_model()