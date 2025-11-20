# 🏭 Visual Quality Control System (End-to-End MLOps)

An automated visual inspection system for manufacturing lines, capable of detecting product defects (e.g., broken bottles) in real-time. Built with **YOLOv8**, **ONNX Runtime**, and **MLflow**, simulating an Edge AI deployment workflow.

![Demo Screenshot]("C:\Visual_qc_project\Bad.jpg")("C:\Visual_qc_project\Good.jpg") *<-- Add your screenshot here!*

## 🚀 Key Features
* **Defect Detection:** Custom YOLOv8 model trained to detect glass anomalies.
* **Edge Optimization:** Model quantized and exported to **ONNX** for sub-100ms inference on CPU.
* **MLOps Pipeline:** Automated experiment tracking and model registry using **MLflow**.
* **Interactive Dashboard:** Streamlit app for factory operator simulation.

## 🛠️ Tech Stack
* **Model:** YOLOv8 (Ultralytics), PyTorch
* **Serving:** ONNX Runtime, Streamlit
* **Tracking:** MLflow
* **Data:** MVTec AD Dataset (Bottle Category)

## 📊 Results
* **mAP@50:** 78.9%
* **Inference Speed:** ~85ms (CPU)

## 💻 How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run src/demo_app.py`
