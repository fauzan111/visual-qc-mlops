# src/serve_model.py

import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import json
import uvicorn

# --- Global Initialization (Run only once on startup) ---
# NOTE: Replace with the actual path to your exported ONNX model
ONNX_MODEL_PATH = "./artifacts/best.onnx" 
# OR load dynamically from MLflow artifacts path

try:
    # 1. Initialize ONNX Runtime Session (High performance inference engine)
    sess_options = ort.SessionOptions()
    # Add optimization flags if needed (e.g., enable graph optimization)
    session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=sess_options)
    print(f"ONNX Model loaded successfully: {ONNX_MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Could not load ONNX model. Is the path correct? {e}")
    session = None

app = FastAPI(title="Edge QC Inference Service")

# --- Helper Functions (YOLO Pre/Post-processing) ---

def preprocess_image(image: Image.Image, img_size=(640, 640)):
    """Resizes and normalizes the image for YOLOv8 input."""
    image = image.convert("RGB").resize(img_size)
    # Convert to numpy array, scale, transpose (HWC -> CHW), add batch dim
    input_array = np.array(image, dtype=np.float32) / 255.0
    input_array = input_array.transpose(2, 0, 1)  # HWC to CHW
    input_tensor = np.expand_dims(input_array, 0) # Add batch dimension
    return input_tensor

def postprocess_yolo_output(output, original_width, original_height, threshold=0.25):
    """Parses raw YOLO output tensor into readable bounding boxes and scores."""
    # (Implementation of NMS and coordinate scaling is complex and omitted here for brevity,
    # but in a real project, this is where the logic for non-max suppression and 
    # resizing back to original coordinates would go.)
    
    # Example: filter by confidence threshold
    predictions = []
    
    # Simplified structure: assuming output is a single tensor of [1, 84, N]
    # where N is the number of boxes, and rows are [box, score, class]
    # This is highly dependent on the exact YOLOv8 export format.
    
    # For now, return the raw array structure as a placeholder.
    # In a real scenario, you would implement the official NMS/decode logic here.
    
    # Let's just return a placeholder for the coordinates and score
    # In a real deployment, the output needs to be scaled back to the original image size
    return {"status": "Post-processing Placeholder", "raw_output_shape": str(output.shape)}

# --- API Endpoints ---

@app.get("/health")
def health_check():
    """Simple health check to confirm the service is running."""
    return {"status": "ok", "model_loaded": session is not None}

@app.post("/predict")
async def predict_defect(file: UploadFile = File(...)):
    """Performs inference on the uploaded image file."""
    if session is None:
        return {"error": "Model not loaded. Check server startup logs."}, 503
        
    try:
        # Read image data from the uploaded file
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        
        original_width, original_height = image.size

        # 1. Preprocessing
        input_tensor = preprocess_image(image)
        
        # 2. Inference with ONNX Runtime
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Run inference
        outputs = session.run([output_name], {input_name: input_tensor})
        
        # 3. Post-processing (YOLO decode)
        predictions = postprocess_yolo_output(
            outputs[0], 
            original_width, 
            original_height
        )

        return {
            "filename": file.filename,
            "predictions": predictions,
            "model_version": "v1.0-onnx"
        }
        
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}, 500

# To run the API: uvicorn src.serve_model:app --host 0.0.0.0 --port 80