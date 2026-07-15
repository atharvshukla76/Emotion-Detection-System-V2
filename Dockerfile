FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential libsndfile1 ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir noisereduce numba
COPY . .
# Pre-download Hugging Face models during the build phase to prevent 60-second timeout crashes on startup
RUN python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-tiny.en'); pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', top_k=None); pipeline('image-classification', model='trpakov/vit-face-expression', top_k=None)"
EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
