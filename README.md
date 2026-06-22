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
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?logo=TensorFlow&logoColor=white)](https://tensorflow.org)
</div>

<br>

Welcome to the **Emotion Detection System V2**, a highly advanced, real-time artificial intelligence application. 

While traditional emotion detection systems rely on just a single input (usually a static image of a face), this system utilizes a groundbreaking **Quad-Modal Architecture**. By seamlessly fusing context, tone, facial motion, and static expressions, the system guarantees extremely robust predictions even in the most difficult edge cases (like silent webcams or sarcasm).

---

## 🚀 Live Demo

Experience the Quad-Modal fusion in real-time directly on Hugging Face Spaces:  
👉 **[Launch Emotion Detection System V2](https://huggingface.co/spaces/AtharvShukla/Emotion-Detection-System-V2)**

---

## 🧠 The Quad-Modal Architecture

This system acts as a digital brain, employing four distinct neural networks working in perfect harmony:

### 1. Static FER Vision (Facial Expression Recognition)
* **Model:** `dima806/facial_emotions_image_detection` (Transformers Pipeline)
* **Purpose:** Analyzes the physical geometry of your expression from a high-quality static frame (smiles, frowns, scrunched noses).
* **Why it matters:** Guarantees flawless predictions when the user is completely silent and motionless, serving as the ultimate baseline for "Vision-Only" mode.

### 2. Dynamic RAVDESS Vision (Optical Flow)
* **Model:** Custom 3D CNN
* **Purpose:** Analyzes the temporal motion of the face (head bobbing, jaw movement, rapid blinking) over 15 consecutive frames.
* **Why it matters:** Captures the physical intensity and dynamic energy of an emotion that a static photograph simply cannot see.

### 3. Dynamic RAVDESS Audio (Tonality & Pitch)
* **Model:** Custom 1D CNN + Global Average Pooling
* **Purpose:** Analyzes the Mel-spectrogram of the user's voice to detect pitch variations, shouting, whispering, or shaking.
* **Why it matters:** Catches complex emotions like "Fear" or "Anger" that are heavily projected through vocal tone rather than facial expressions.

### 4. Linguistic NLP (Meaning & Context)
* **Models:** `openai/whisper-tiny.en` (Speech-to-Text) + `j-hartmann/emotion-english-distilroberta-base` (Text Emotion)
* **Purpose:** Transcribes the user's speech in real-time and analyzes the actual semantic meaning of the words.
* **Why it matters:** Solves the "Sarcasm" problem. If a user maintains a completely stoic face but says *"I am so sad right now"*, the NLP network intelligently overpowers the vision network to output the true emotion.

---

## ⚖️ The Fusion Engine (How it thinks)

The core intelligence of the system lies in how it dynamically weights the outputs of all four models based on environmental context:

* **Vision-Only Mode (Silence Detected):**
  If the smart noise-gate detects silence, the system recognizes that the temporal motion and audio networks lack data. It automatically shifts **70% of the voting power** to the Static FER Vision model, perfectly reading silent expressions without defaulting to noise.
  
* **Voice + Meaning Mode (Speech Detected):**
  When speech is detected, the system fuses all four models. If the NLP model is highly confident about the meaning of your words, it dynamically boosts its own voting power (up to 40%). The Static Image, Temporal Video, and Audio models split the remaining 60%, creating a foolproof, unified prediction.

---

## 💻 Running Locally

### Prerequisites
* Python 3.9 or higher
* A functional webcam and microphone

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
   uvicorn api:app --host 0.0.0.0 --port 7860
   ```

4. **Access the application:**  
   Open your browser and navigate to `http://localhost:7860`

---

## 🛠️ Built With
* **TensorFlow / Keras:** Custom Audio/Visual network architecture
* **Hugging Face Transformers:** Whisper (ASR), DistilRoberta (NLP), and ViT (FER)
* **FastAPI:** High-performance asynchronous Python backend
* **OpenCV:** Real-time facial extraction and optical flow processing
* **Librosa:** Audio mel-spectrogram extraction

---

<div align="center">
  <h3>Architected and Developed by</h3>
  <h2><b>Atharv Shukla</b></h2>
  <p><i>Pushing the boundaries of Human-Computer Interaction through multimodal Artificial Intelligence.</i></p>
</div>
