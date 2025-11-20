# Use lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "src/demo_app.py", "--server.address=0.0.0.0"]