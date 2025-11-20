import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- Configuration ---
# Path to the ONNX model we just exported
MODEL_PATH = r"artifacts/best.onnx"

st.set_page_config(page_title="Visual QC Inspector", page_icon="🔍")

st.title("🏭 Visual Quality Control System")
st.subheader("Real-time Defect Detection (ONNX Edge Simulation)")

# --- Sidebar ---
st.sidebar.header("System Status")
if os.path.exists(MODEL_PATH):
    st.sidebar.success(f"Model Loaded: {os.path.basename(MODEL_PATH)}")
    try:
        # Load the ONNX model using Ultralytics (it auto-detects the engine)
        model = YOLO(MODEL_PATH, task="detect")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        model = None
else:
    st.sidebar.error("Model not found! Check path.")
    model = None

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# --- Main Interface ---
uploaded_file = st.file_uploader("Upload a Product Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    # 1. Display Original Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Inspection"):
        with st.spinner("Running Neural Network Inference..."):
            # 2. Run Inference
            # We run prediction on the image using the ONNX model
            results = model.predict(image, conf=conf_threshold)
            
            # 3. Visualize Results
            # Ultralytics provides a method to plot results on the image
            res_plotted = results[0].plot()
            
            # Convert BGR (OpenCV) to RGB (Streamlit)
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # 4. Show Output
            st.image(res_plotted_rgb, caption="Defect Detection Result", use_container_width=True)
            
            # 5. Show Metrics
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.error(f"⚠️ DEFECT DETECTED: Found {len(boxes)} anomalies.")
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    st.write(f"- Defect Type: {results[0].names[cls_id]} (Confidence: {conf:.2f})")
            else:
                st.success("✅ PASSED: No defects detected.")

else:
    st.info("Please upload an image to begin inspection.")