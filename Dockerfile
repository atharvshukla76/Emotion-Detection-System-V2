FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Ensure Haar Cascade exists for face detection
RUN CASCADE_DIR=$(python -c "import cv2; print(cv2.data.haarcascades)") && \
    mkdir -p "$CASCADE_DIR" && \
    curl -fsSL -o "${CASCADE_DIR}haarcascade_frontalface_default.xml" \
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
COPY . .
EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
