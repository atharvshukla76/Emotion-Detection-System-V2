import os
import cv2
import pickle
import shutil
import tempfile
import threading
import subprocess
import traceback
import numpy as np
import librosa
import noisereduce as nr
from transformers import pipeline
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from PIL import Image
import uuid

# ==========================================
# 1. GLOBAL CONFIGURATION & STATE
# ==========================================
prediction_tasks = {}

SR = 22050
DURATION = 3
SAMPLES = SR * DURATION
N_MELS = 96
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
MAX_FRAMES = 150

TARGET_AUDIO_SHAPE = (150, 136, 1)
TARGET_VIDEO_SHAPE = (64, 64, 30)
MODEL_DIR = "saved_model"

app = FastAPI(title="Moodwave V2.0 Quadra-Modal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared AI Models
model = None
encoder = None
mean = None
std = None
vid_mean = None
vid_std = None
whisper_pipe = None
text_emotion_pipe = None
fer_pipe = None
face_cascade = None

# Async Buffers
last_known_text_probs = None
last_known_transcription = ""

# ==========================================
# 2. LIFECYCLE & MODEL LOADING
# ==========================================
@app.on_event("startup")
def load_resources():
    global model, encoder, mean, std, vid_mean, vid_std
    global whisper_pipe, text_emotion_pipe, fer_pipe, face_cascade
    
    print("[STARTUP] Initializing Quadra-Modal Engine...")
    
    # Load Main AV Model
    model_path = os.path.join(MODEL_DIR, "multimodal_emotion_model.keras")
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        with open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb") as f:
            encoder = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "norm.pkl"), "rb") as f:
            norm = pickle.load(f)
            mean, std = norm["mean"], norm["std"]
            vid_mean, vid_std = norm.get("vid_mean"), norm.get("vid_std")
    else:
        print("[WARNING] Multimodal model not found! Prediction will fail.")

    # Load NLP Models
    whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
    text_emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    
    # Load FER Vision Model
    fer_pipe = pipeline("image-classification", model="trpakov/vit-face-expression", top_k=None)
    
    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("[STARTUP] All systems online.")

# ==========================================
# 3. PREPROCESSING PIPELINES
# ==========================================
def preprocess_audio(file_path):
    try:
        signal, _ = librosa.load(file_path, sr=SR)
        std_sig = np.std(signal)
        whisper_signal = None
        
        # Voice Activity Detector (VAD)
        if len(signal) == 0 or std_sig < 0.001:
            print(f"[DEBUG] VAD: Silence Detected (std: {std_sig:.5f}).")
        else:
            try:
                whisper_signal = nr.reduce_noise(y=signal, sr=SR, prop_decrease=0.75)
            except Exception:
                whisper_signal = signal

        # Format for Custom AV Model
        trimmed, index = librosa.effects.trim(signal, top_db=30)
        start_offset = index[0] if len(trimmed) > 0 else 0
        signal = trimmed if len(trimmed) > 0 else signal
        signal = signal - np.mean(signal)
        
        if len(signal) > SAMPLES:
            best_start = (len(signal) - SAMPLES) // 2 if len(signal)/SR <= 4.5 else 0
            signal = signal[best_start:best_start+SAMPLES]
        else:
            signal = np.pad(signal, (0, SAMPLES - len(signal)))
            
        t_start = (start_offset) / SR
        mel = librosa.feature.melspectrogram(y=signal, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel)
        mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features = np.concatenate((mel_db, mfcc), axis=0).T
        
        features = features[:MAX_FRAMES, :] if features.shape[0] > MAX_FRAMES else np.pad(features, ((0, MAX_FRAMES - features.shape[0]), (0, 0)))
        features = np.clip(np.nan_to_num(features), -100.0, 100.0)
        
        is_silent = bool(len(signal) == 0 or std_sig < 0.005)
        return np.expand_dims(features, axis=-1), t_start, whisper_signal, is_silent
    except Exception as e:
        print(f"[ERROR] Audio processing: {e}")
        return np.zeros(TARGET_AUDIO_SHAPE, dtype=np.float32), 0.0, None, True

def preprocess_video(video_path, t_start=0.0):
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        start_frame = int(t_start * fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame >= total_frames: start_frame = max(0, total_frames - 15)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        fer_frames = []
        face_detected_count = 0
        brightness_list = []
        
        for _ in range(15):
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            if len(faces) > 0:
                face_detected_count += 1
                x, y, w, h = max(faces, key=lambda f: f[2]*f[3]) # Largest face
                face_crop = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_crop, (64, 64))
                frames.append(face_resized)
                fer_frames.append(frame[y:y+h, x:x+w]) # Save RGB crop for FER
                
                brightness_list.append(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:,:,0]))
            else:
                h, w = gray.shape
                center_crop = gray[h//4:3*h//4, w//4:3*w//4]
                frames.append(cv2.resize(center_crop, (64, 64)))
        cap.release()
        
        if not frames: return np.zeros(TARGET_VIDEO_SHAPE, dtype=np.float32), False, [], 0, 0.0, 100.0
        
        # Calculate Optical Flow for AV Model
        flow_frames = []
        mouth_flow_vals = []
        for i in range(len(frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(frames[i], frames[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_frames.append(mag)
            
            # Estimate mouth region flow (bottom 1/3 of the 64x64 face crop)
            mouth_region = mag[42:64, 16:48]
            mouth_flow_vals.append(np.mean(mouth_region))
            
        flow_frames.append(np.zeros_like(frames[0], dtype=np.float32))
        mouth_variance = float(np.mean(mouth_flow_vals)) if mouth_flow_vals else 0.0
        
        vid_features = np.zeros((64, 64, 30), dtype=np.float32)
        for i in range(len(frames)):
            vid_features[:, :, i*2] = frames[i]
            vid_features[:, :, i*2 + 1] = flow_frames[i]
            
        motion_mean = float(np.mean(np.array(flow_frames)))
        mean_brightness = float(np.mean(brightness_list)) if brightness_list else 100.0
        
        is_face_visible = face_detected_count > 0
        return np.expand_dims(vid_features, axis=0), is_face_visible, fer_frames, face_detected_count, motion_mean, mean_brightness, mouth_variance
    except Exception as e:
        print(f"[ERROR] Video processing: {e}")
        return np.zeros(TARGET_VIDEO_SHAPE, dtype=np.float32), False, [], 0, 0.0, 100.0, 0.0

# ==========================================
# 4. ASYNCHRONOUS NLP THREAD
# ==========================================
def run_nlp_async(signal_16k):
    global last_known_text_probs, last_known_transcription
    try:
        out = whisper_pipe({"raw": signal_16k, "sampling_rate": 16000})
        text = out.get("text", "").strip()
        
        # Hallucination Guard
        hals = ["thank you", "subscribe", "thanks for watching", "by subtitlr", ""]
        is_hal = any(h == text.lower().strip(" .!,") for h in hals)
        
        if text and not is_hal:
            print(f"[DEBUG Async] Transcription: {text}")
            raw_res = text_emotion_pipe(text, top_k=None, truncation=True, max_length=512)
            
            # Safely unpack the pipeline output which varies between Hugging Face versions
            while isinstance(raw_res, list) and len(raw_res) > 0 and isinstance(raw_res[0], list):
                raw_res = raw_res[0]
            emotions = [raw_res] if isinstance(raw_res, dict) else raw_res
            
            t_probs = np.zeros(len(encoder.classes_), dtype=np.float32)
            lmap = {"joy":"Happy", "sadness":"Sad", "anger":"Angry", "fear":"Fear", "disgust":"Disgust", "neutral":"Neutral", "surprise":"Neutral"}
            
            for res in emotions:
                if isinstance(res, dict) and 'label' in res:
                    target = lmap.get(res['label'])
                    if target in encoder.classes_:
                        t_probs[int(np.where(encoder.classes_ == target)[0][0])] += res.get('score', 0.0)
                    
            last_known_text_probs = t_probs
            last_known_transcription = text
    except Exception as e:
        print(f"[DEBUG] NLP Thread Error: {e}")

# ==========================================
# 5. CORE PREDICTION & FUSION ENGINE
# ==========================================
def process_prediction_task(task_id: str, temp_dir: str, video_path: str, audio_path: str):
    global last_known_text_probs, last_known_transcription
    try:
        
        # 1. Extract Features
        aud_feat, t_start, clean_sig, aud_silent = preprocess_audio(audio_path)
        vid_feat, vid_active, fer_frames, face_count, m_mean, v_bright, mouth_variance = preprocess_video(video_path, t_start)
        
        # Lip-Sync Diarization (Active Speaker Detection)
        if vid_active and not aud_silent and mouth_variance < 0.15:
            print(f"[DEBUG] Lip-Sync Mismatch (Mouth Flow: {mouth_variance:.3f}). Speaker is off-camera. Muting background audio.")
            aud_silent = True
        
        # Normalize Data
        if mean is not None and std is not None:
            m_a = np.array(mean, dtype=np.float32)[..., np.newaxis] if np.array(mean).ndim == 2 else mean
            s_a = np.array(std, dtype=np.float32)[..., np.newaxis] if np.array(std).ndim == 2 else std
            aud_feat = (aud_feat - m_a) / (s_a + 1e-6)
        if vid_mean is not None and vid_std is not None:
            vid_feat = (vid_feat - vid_mean) / (vid_std + 1e-6)
            
        aud_feat = np.expand_dims(np.reshape(aud_feat, (150, 136, 1)), axis=0)
        
        # 2. Trigger Async NLP
        if not aud_silent and clean_sig is not None:
            sig_16k = librosa.resample(y=clean_sig, orig_sr=SR, target_sr=16000).astype(np.float32)
            nlp_thread = threading.Thread(target=run_nlp_async, args=(sig_16k,))
            nlp_thread.start()
            # Asynchronous State Buffer: We DO NOT join the thread. 
            # The API will retrieve the last known NLP state and return instantly, 
            # while Whisper updates the global buffer for the next inference frame.
            
        # 3. Retrieve Probabilities
        probs_av = model.predict({"audio_input": aud_feat, "video_input": vid_feat}, verbose=0)[0]
        
        probs_fer = np.zeros_like(probs_av)
        if vid_active and len(fer_frames) > 0:
            lmap = {"happy":"Happy", "sad":"Sad", "angry":"Angry", "fear":"Fear", "disgust":"Disgust", "neutral":"Neutral", "surprise":"Neutral",
                    "Happy":"Happy", "Sad":"Sad", "Angry":"Angry", "Fear":"Fear", "Disgust":"Disgust", "Neutral":"Neutral", "Surprise":"Neutral"}
            for frame in fer_frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fer_pipe(Image.fromarray(rgb), top_k=None)
                for r in res:
                    t = lmap.get(r['label'])
                    if t in encoder.classes_: probs_fer[int(np.where(encoder.classes_ == t)[0][0])] += r['score']
            probs_fer = probs_fer / (np.sum(probs_fer) + 1e-6)
            
        probs_text = last_known_text_probs if last_known_text_probs is not None else np.zeros_like(probs_av)
        text_str = last_known_transcription if last_known_transcription else "[Silence]"
        
        # ==========================================
        # 6. DYNAMIC QUAD-MODAL FUSION (Algorithm 1)
        # ==========================================
        has_fer = np.sum(probs_fer) > 0
        has_text = np.sum(probs_text) > 0
        
        # Algorithm 1 Base Implementation
        if not aud_silent:
            wt = 0.4 if has_text and np.max(probs_text) > 0.7 else 0.3
            if not has_text: wt = 0.0
            wf = 0.3 if has_fer else 0.0
            wav = 1.0 - wt - wf
        else:
            wt = 0.0
            wf = 0.7 if has_fer else 0.0
            wav = 0.3 if has_fer else 1.0
            
        # Sarcasm / Contradiction Override
        if has_fer and has_text and not aud_silent:
            em_face = encoder.classes_[int(np.argmax(probs_fer))]
            em_text = encoder.classes_[int(np.argmax(probs_text))]
            em_tone = encoder.classes_[int(np.argmax(probs_av))]
            
            def conflict(e1, e2):
                p, n = ["Happy"], ["Angry", "Disgust", "Sad", "Fear"]
                return (e1 in p and e2 in n) or (e1 in n and e2 in p)
                
            if conflict(em_face, em_text) or conflict(em_face, em_tone):
                print("[DEBUG] Sarcasm! Facial Truth overrides Semantics.")
                wt = 0.0; wav = 0.2; wf = 0.8
                
        # Environmental Shields
        if has_fer:
            if m_mean > 2.0:
                print("[DEBUG] Shield: Camera Shake. Trusting Static ViT.")
                wf = 0.85; wav = 0.15; wt = 0.0
            elif v_bright < 80:
                print("[DEBUG] Shield: Dim Lighting. Trusting AV Flow.")
                wf = 0.15; wav = 0.85; wt = 0.0

        # Math Fusion
        total_w = wf + wav + wt
        if total_w > 0: wf /= total_w; wav /= total_w; wt /= total_w
        else: wav = 1.0
        
        print(f"[DEBUG] Weights -> Face: {wf:.2f}, AV/Tone: {wav:.2f}, Text: {wt:.2f}")
        final_probs = (probs_av * wav) + (probs_fer * wf) + (probs_text * wt)
        final_probs = final_probs / (np.sum(final_probs) + 1e-6)
        
        # Micro-Emotion Amplifier
        n_idx = int(np.where(encoder.classes_ == 'Neutral')[0][0])
        if np.argmax(final_probs) == n_idx:
            srt = np.argsort(final_probs)[::-1]
            if final_probs[srt[1]] > 0.15:
                print(f"[DEBUG] Micro-Expression Amplifier triggered for {encoder.classes_[srt[1]]}")
                final_probs[n_idx] *= 0.35
                final_probs = final_probs / (np.sum(final_probs) + 1e-6)
                
        final_idx = int(np.argmax(final_probs))
        pred_label = encoder.inverse_transform([final_idx])[0]
        
        prediction_tasks[task_id] = {
            "status": "completed",
            "result": {
                "time": str(datetime.now()),
                "emotion": pred_label,
                "confidence": float(final_probs[final_idx]),
                "all_scores": {encoder.inverse_transform([i])[0]: float(p) for i, p in enumerate(final_probs)},
                "transcription": text_str,
                "diagnostics": {
                    "face_detected_frames": face_count,
                    "audio_silent": aud_silent,
                    "video_active": vid_active
                }
            }
        }
    except Exception as e:
        print(f"[FATAL ERROR IN THREAD]: {e}")
        traceback.print_exc()
        prediction_tasks[task_id] = {"status": "failed", "error": str(e)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ==========================================
# 7. FASTAPI ENDPOINTS
# ==========================================
@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("moodwave.html"):
        with open("moodwave.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return "<h1>Moodwave Quadra-Modal API is Online. (GUI not found)</h1>"

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")
        
    task_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, file.filename)
    audio_path = os.path.join(temp_dir, "audio.wav")
    
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1', audio_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        print("[WARNING] ffmpeg failed to extract audio. Proceeding as silent video.")
        open(audio_path, 'a').close()

    prediction_tasks[task_id] = {"status": "processing"}
    threading.Thread(target=process_prediction_task, args=(task_id, temp_dir, video_path, audio_path)).start()
    return {"task_id": task_id}

@app.get("/result/{task_id}")
def get_result(task_id: str):
    if task_id not in prediction_tasks:
        raise HTTPException(status_code=404, detail="Task ID not found")
    res = prediction_tasks[task_id]
    if res["status"] == "completed" or res["status"] == "failed":
        del prediction_tasks[task_id]
    return res