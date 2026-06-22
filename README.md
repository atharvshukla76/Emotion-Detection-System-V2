---
title: Emotion Detection System V2
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🧠 Emotion Detection System V2 (Quad-Modal Fusion)

Welcome to the **Emotion Detection System V2**, a state-of-the-art AI application that detects human emotion by fusing four different data streams in real-time. 

While traditional emotion detection systems rely on just a single input (usually a static image of a face), this system utilizes a **Quad-Modal Architecture** to perfectly understand context, tone, motion, and expression, guaranteeing extremely robust predictions even in tricky edge cases (like silent webcams or sarcasm).

## 🚀 Live Demo
You can try the live version of this application directly on Hugging Face Spaces:
**[Launch Emotion Detection System V2](https://huggingface.co/spaces/AtharvShukla/Emotion-Detection-System-V2)**

---

## 🏗️ The Quad-Modal Architecture

This system uses four distinct neural networks working in harmony:

1. **Static FER Vision (Facial Expression Recognition)**
   * **Model:** `dima806/facial_emotions_image_detection` (Transformers Pipeline)
   * **Purpose:** Grabs a high-quality static frame of the user's face and analyzes the physical shape of their expression (smiles, frowns, scrunched noses).
   * **Why it matters:** Guarantees perfect predictions when the user is completely silent and motionless, acting as the primary baseline for "Vision-Only" mode.

2. **Dynamic RAVDESS Vision (Optical Flow)**
   * **Model:** Custom 3D CNN (Trained on RAVDESS)
   * **Purpose:** Analyzes the temporal motion of the face (head bobbing, jaw movement, rapid blinking) over 15 consecutive frames.
   * **Why it matters:** Captures the physical intensity and dynamic energy of an emotion that a static image would miss.

3. **Dynamic RAVDESS Audio (Tonality & Pitch)**
   * **Model:** Custom 1D CNN + Global Average Pooling (Trained on RAVDESS)
   * **Purpose:** Analyzes the Mel-spectrogram of the user's voice to detect pitch variations, shouting, whispering, or shaking in the voice.
   * **Why it matters:** Catches emotions like "Fear" or "Anger" that are heavily projected through vocal tone rather than just facial expressions.

4. **Linguistic NLP (Meaning & Context)**
   * **Models:** `openai/whisper-tiny.en` (Speech-to-Text) + `j-hartmann/emotion-english-distilroberta-base` (Text Emotion)
   * **Purpose:** Transcribes what the user is saying in real-time and analyzes the actual semantic meaning of the words.
   * **Why it matters:** Solves the "Sarcasm" or "Stoic" problems. If a user has a completely neutral face but says "I am so sad right now", the NLP network intelligently overpowers the vision network to output the correct emotion.

---

## ⚙️ How the Fusion Math Works (The "Brain")

The outputs of all four models are dynamically weighted based on the current context:

* **Vision-Only Mode (Silence Detected):**
  If the noise gate detects that the user is silent, the system recognizes that the RAVDESS motion-tracking network and Audio network will fail. It automatically shifts **70% of the voting power** to the Static FER Vision model, perfectly reading the silent expression.
  
* **Voice + Meaning Mode (Speech Detected):**
  When the user is speaking, the system uses all four models. If the NLP model is highly confident about the meaning of the words (e.g., probability > 0.7), it dynamically boosts its own voting power (up to 40%), while the Static Image, Temporal Video, and Audio models split the remaining 60%. This creates a completely foolproof, unified prediction.

---

## 💻 Running Locally

### Prerequisites
* Python 3.9 or higher
* A webcam and microphone

### Installation

1. Clone the repository:
```bash
git clone https://github.com/atharvshukla76/Emotion-Detection-System.git
cd Emotion-Detection-System
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the FastAPI server:
```bash
uvicorn api:app --host 0.0.0.0 --port 7860
```

4. Open your browser and navigate to:
`http://localhost:7860`

---

## 🛠️ Built With
* **TensorFlow / Keras:** Custom Audio/Visual network architecture
* **Hugging Face Transformers:** Whisper (ASR), DistilRoberta (NLP), and ViT (FER)
* **FastAPI:** High-performance asynchronous Python backend
* **OpenCV:** Real-time facial extraction and optical flow processing
* **Librosa:** Audio mel-spectrogram extraction
