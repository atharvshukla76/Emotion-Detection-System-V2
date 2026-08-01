---
title: Emotion Detection System V2
emoji: 🎭
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

<div align="center">
  <h1>🎭 Emotion Detection System V2</h1>
  <p><b>A state-of-the-art Quad-Modal AI Architecture for Real-Time Human Emotion Analysis</b></p>
  
  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AtharvShukla/Emotion-Detection-System-V2)
  [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
</div>

<br>

Welcome to the **Emotion Detection System V2**, a highly advanced, real-time artificial intelligence application. 

While traditional emotion detection systems rely on just a single input (usually a static image of a face), this system utilizes a groundbreaking **Quad-Modal Architecture** trained on the RAVDESS and CREMA-D datasets. By seamlessly fusing context, tone, facial motion, and static expressions, the system guarantees extremely robust predictions across 6 emotional classes: **Angry, Disgust, Fear, Happy, Neutral, and Sad**.

---

## 🚀 Live Demo

Experience the Quad-Modal fusion in real-time directly on Hugging Face Spaces:  
👉 **[Launch Emotion Detection System V2](https://huggingface.co/spaces/AtharvShukla/Emotion-Detection-System-V2)**

---

## 🧠 The Quad-Modal Architecture

This system acts as a digital brain, employing four distinct neural networks working in perfect harmony:

### 1. Static FER Vision (Facial Expression Recognition)
* **Model:** `dima806/facial_emotions_image_detection` (ViT Transformer Pipeline)
* **Purpose:** Analyzes the physical geometry of your expression from a high-quality static frame (smiles, frowns, scrunched noses).
* **Why it matters:** Guarantees flawless predictions when the user is completely silent and motionless, serving as the ultimate baseline for "Vision-Only" mode.

### 2. Dynamic Audio-Visual Engine (Phase 2 PyTorch Architecture)
* **Models:** PyTorch Custom MobileNetV2 + SE Blocks & Multihead Attention
* **Purpose (Vision):** Uses **MediaPipe Face Mesh** to extract a tight ROI of the eyes and mouth, then computes 2-channel dense optical flow over a 15-frame rolling window.
* **Purpose (Audio):** Computes precise Mel-spectrograms strictly aligned with the exact training distribution (zero z-score normalization, pure [-100, 100] signal clipping).
* **Why it matters:** Captures the physical intensity of dynamic facial energy and complex vocal tones simultaneously through an advanced PyTorch cross-modal attention layer.

### 3. Linguistic NLP (Meaning & Context)
* **Models:** `openai/whisper-base.en` (Speech-to-Text) + `j-hartmann/emotion-english-distilroberta-base` (Text Emotion)
* **Purpose:** Transcribes the user's speech entirely synchronously and analyzes the actual semantic meaning of the words.
* **Why it matters:** Solves the "Sarcasm" problem. If a user maintains a completely stoic face but says *"I am so sad right now"*, the NLP network intelligently overpowers the vision network to output the true emotion.

---

## ⚖️ The Fusion Engine (How it thinks)

The core intelligence of the system lies in how it dynamically weights the outputs of all models based on environmental context:

* **Synchronous Modality Alignment:**
  By moving the NLP transcription into the synchronous request thread, the system ensures zero data leakage between concurrent API requests. Every prediction perfectly aligns the optical flow, spectrogram, and transcribed text.

* **Targeted Neutral Dampening:**
  When a confident runner-up emotion is detected alongside high dynamic optical flow, the fusion engine intelligently down-weights the "Neutral" class. This resolves the common machine-learning trap of defaulting to Neutral during complex expressions.

* **Temporal ROI Consistency:**
  If the MediaPipe face tracker temporarily loses tracking due to lighting or head turns, the pipeline automatically inherits the previous frame's bounding box. This prevents erratic optical flow vectors and perfectly mimics the structural consistency of the training pipeline.

---

## 💻 Running Locally

### Prerequisites
* Python 3.9 or higher
* A functional webcam and microphone
* ffmpeg (required for audio processing)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/atharvshukla76/Emotion-Detection-System-V2.git
   cd Emotion-Detection-System-V2
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the FastAPI server:**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

4. **Access the application:**  
   Open your browser and navigate to `http://localhost:8000`

---

## 🛠️ Built With
* **PyTorch:** Phase 2 Audio/Visual cross-modal attention network
* **Hugging Face Transformers:** Whisper-base (ASR), DistilRoberta (NLP), and ViT (FER)
* **FastAPI:** High-performance synchronous Python backend
* **MediaPipe & OpenCV:** Real-time facial extraction and optical flow processing
* **Librosa:** Audio mel-spectrogram extraction

---

<div align="center">
  <h3>Architected and Developed by</h3>
  <h2><b>Atharv Shukla</b></h2>
  <p><i>Pushing the boundaries of Human-Computer Interaction through multimodal Artificial Intelligence.</i></p>
</div>
