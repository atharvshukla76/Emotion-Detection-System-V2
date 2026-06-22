import os
import cv2
import pickle
import shutil
import tempfile
import subprocess
import traceback
import numpy as np
import librosa
import noisereduce as nr
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# --- CONFIGURATION (Must match training constants) ---
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

app = FastAPI(title="Moodwave V2.0 Multimodal API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://atharvshukla76.github.io",
        "https://emotion-detection-system-1-ycpg.onrender.com",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global resource holders
model = None
encoder = None
mean = None
std = None
vid_mean = None
vid_std = None

@app.on_event("startup")
def load_resources():
    global model, encoder, mean, std, vid_mean, vid_std
    try:
        model_path = os.path.join(MODEL_DIR, "multimodal_emotion_model.keras")
        encoder_path = os.path.join(MODEL_DIR, "encoder.pkl")
        norm_path = os.path.join(MODEL_DIR, "norm.pkl")
        
        if not os.path.exists(model_path):
            print(f"[STARTUP WARNING] Multimodal model not found at {model_path}. You need to run training first to generate it.")
            return
            
        model = tf.keras.models.load_model(model_path)
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
        with open(norm_path, "rb") as f:
            norm = pickle.load(f)
            mean = norm["mean"]
            std = norm["std"]
            vid_mean = norm.get("vid_mean")
            vid_std = norm.get("vid_std")
            
        print("[STARTUP] Multimodal resources successfully loaded.")
        print(f"[STARTUP] Classes: {list(encoder.classes_)}")
        print(f"[STARTUP] Video normalization: {'loaded' if vid_mean is not None else 'not found'}")
    except Exception as e:
        print(f"[STARTUP ERROR] Resource load failed: {e}")

# =====================================================================
# 🔊 AUDIO PREPROCESSING
# =====================================================================
def preprocess_audio(file_path):
    try:
        signal, _ = librosa.load(file_path, sr=SR)
        
        # Apply Spectral Gating Noise Reduction to match studio training conditions
        if len(signal) > 0:
            signal = nr.reduce_noise(y=signal, sr=SR, prop_decrease=0.85)
            
        if len(signal) == 0 or np.std(signal) < 0.015:
            return np.zeros(TARGET_AUDIO_SHAPE, dtype=np.float32), 0.0
            
        # Trim silent boundaries from outer edges
        trimmed, index = librosa.effects.trim(signal, top_db=30)
        start_offset = 0
        if len(trimmed) > 0:
            signal = trimmed
            start_offset = index[0]
            
        signal = signal - np.mean(signal)
        
        duration = len(signal) / SR
        best_start = 0
        if len(signal) > SAMPLES:
            if duration <= 4.5:
                # Center crop for short dataset files to match training
                best_start = (len(signal) - SAMPLES) // 2
            else:
                # Find the 3-second window with the highest rolling energy for live recordings
                signal_sq = signal**2
                cumsum = np.cumsum(signal_sq)
                max_energy = -1
                hop = 1024
                for start in range(0, len(signal) - SAMPLES, hop):
                    energy = cumsum[start + SAMPLES] - cumsum[start]
                    if energy > max_energy:
                        max_energy = energy
                        best_start = start
            signal = signal[best_start:best_start+SAMPLES]
        else:
            signal = np.pad(signal, (0, SAMPLES - len(signal)))
            
        # Start time relative to original video file (in seconds)
        t_start = (start_offset + best_start) / SR
            
        mel = librosa.feature.melspectrogram(y=signal, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel)
        
        mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features = np.concatenate((mel_db, mfcc), axis=0).T
        
        if features.shape[0] > MAX_FRAMES:
            features = features[:MAX_FRAMES, :]
        else:
            features = np.pad(features, ((0, MAX_FRAMES - features.shape[0]), (0, 0)))
            
        features = np.nan_to_num(features)
        features = np.clip(features, -100.0, 100.0)
        return np.expand_dims(features, axis=-1), t_start
    except Exception as e:
        print(f"Audio extraction error: {e}")
        return np.zeros(TARGET_AUDIO_SHAPE, dtype=np.float32), 0.0

# =====================================================================
# 🎥 VIDEO PREPROCESSING (OPTICAL FLOW)
# =====================================================================
def preprocess_video(video_path, t_start, target_frames=16, img_size=(64, 64)):
    duration = 0.0
    cap = cv2.VideoCapture(video_path)
    
    # Initialize face cascade detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    raw_frames = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0.0
        
        if duration <= 4.5:
            t_start = 0.0
            start_frame = 0
            end_frame = float('inf')
        else:
            # Align video window to be 3.6s (average RAVDESS length) centered around the 3.0s audio window
            vid_t_start = max(0.0, t_start - 0.3)
            start_frame = int(vid_t_start * fps)
            end_frame = int((vid_t_start + 3.6) * fps)
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx < start_frame:
                frame_idx += 1
                continue
            if frame_idx > end_frame:
                break
                
            raw_frames.append(frame)
            frame_idx += 1
    except Exception as e:
        print(f"Error reading frames: {e}")
    finally:
        cap.release()
        
    # Fallback to entire video if active window is empty
    if len(raw_frames) < 2:
        cap = cv2.VideoCapture(video_path)
        raw_frames = []
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frames.append(frame)
        except Exception:
            pass
        finally:
            cap.release()
            
    if len(raw_frames) < 2:
        return np.zeros(TARGET_VIDEO_SHAPE, dtype=np.float32), 0

    # First pass: collect all detected face boxes across frames
    face_boxes = []
    for frame in raw_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) > 0:
            # Select the largest face detected
            x_f, y_f, w_f, h_f = max(faces, key=lambda f: f[2] * f[3])
            face_boxes.append((x_f, y_f, w_f, h_f))
            
    # Calculate stable box using median coordinates
    if len(face_boxes) > 0:
        face_detected_count = len(face_boxes)
        if duration <= 4.5:
            # Force static crop for dataset files to match training exactly
            stable_box = None
        else:
            avg_x = int(np.median([f[0] for f in face_boxes]))
            avg_y = int(np.median([f[1] for f in face_boxes]))
            avg_w = int(np.median([f[2] for f in face_boxes]))
            avg_h = int(np.median([f[3] for f in face_boxes]))
        avg_x = int(np.median([f[0] for f in face_boxes]))
        avg_y = int(np.median([f[1] for f in face_boxes]))
        avg_w = int(np.median([f[2] for f in face_boxes]))
        avg_h = int(np.median([f[3] for f in face_boxes]))
        stable_box = (avg_x, avg_y, avg_w, avg_h)
    else:
        stable_box = None
        face_detected_count = 0

    # Second pass: crop and stack regions
    processed_frames = []
    for frame in raw_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Use static geometric crop matching the training pipeline exactly
        # The model relies on the background motion (head bobbing relative to background).
        eyes = gray[int(h*0.15):int(h*0.45), int(w*0.2):int(w*0.8)]
        mouth = gray[int(h*0.65):int(h*0.9), int(w*0.25):int(w*0.75)]
            
        if eyes.size == 0 or mouth.size == 0:
            continue
            
        eyes_res = cv2.resize(eyes, (img_size[0], img_size[1] // 2))
        mouth_res = cv2.resize(mouth, (img_size[0], img_size[1] // 2))
        processed_frames.append(np.vstack([eyes_res, mouth_res]))
        
    if len(processed_frames) < 2:
        return np.zeros(TARGET_VIDEO_SHAPE, dtype=np.float32), 0
        
    indices = np.linspace(0, len(processed_frames) - 1, target_frames).astype(int)
    sel_frames = [processed_frames[i] for i in indices]
    
    flow_seq = []
    for i in range(len(sel_frames) - 1):
        prev = sel_frames[i]
        nxt = sel_frames[i+1]
        try:
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            flow = np.clip(flow, -50.0, 50.0)
            flow_seq.append(np.nan_to_num(flow))
        except Exception:
            flow_seq.append(np.zeros((img_size[0], img_size[1], 2), dtype=np.float32))
            
    return np.array(flow_seq, dtype=np.float32), face_detected_count

# =====================================================================
# 🎯 ENDPOINTS
# =====================================================================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("moodwave.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health_check():
    import cv2
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    import os
    cascade_exists = os.path.exists(cascade_path)
    try:
        clf = cv2.CascadeClassifier(cascade_path)
        clf_loaded = not clf.empty()
    except Exception:
        clf_loaded = False
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "cascade_path": cascade_path,
        "cascade_exists": cascade_exists,
        "classifier_loaded": clf_loaded
    }

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Run training and ensure multimodal_emotion_model.keras exists.")
        
    # Create a secure temporary directory to isolate file processing
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "uploaded_capture.mp4")
    audio_path = os.path.join(temp_dir, "extracted_audio.wav")
    
    try:
        # Save video stream
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Extract Audio track via ffmpeg (cross-platform)
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", audio_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # Preprocess Audio
        audio_feat, t_start = preprocess_audio(audio_path)
        print(f"[DEBUG] audio_feat shape after preprocess: {audio_feat.shape}, active window start: {t_start:.2f}s")
        print(f"[DEBUG] mean shape: {np.array(mean).shape}, std shape: {np.array(std).shape}")
        
        # Check if the audio features are empty/silent (all zeros)
        audio_zeros = bool(np.all(audio_feat == 0))
        if not audio_zeros:
            # Normalize — ensure shapes are compatible
            mean_arr = np.array(mean, dtype=np.float32)
            std_arr = np.array(std, dtype=np.float32)
            
            # If mean/std don't have the channel dim, reshape to broadcast correctly
            if mean_arr.ndim == 2:
                mean_arr = mean_arr[..., np.newaxis]  # (150, 136) -> (150, 136, 1)
            if std_arr.ndim == 2:
                std_arr = std_arr[..., np.newaxis]
                
            audio_feat = (audio_feat - mean_arr) / (std_arr + 1e-6)
        
        # Guarantee final shape is (1, 150, 136, 1)
        audio_feat = np.reshape(audio_feat, (150, 136, 1))
        audio_feat = np.expand_dims(audio_feat, axis=0)  # Shape: (1, 150, 136, 1)
        print(f"[DEBUG] audio_feat final shape: {audio_feat.shape}")
        
        # Preprocess Video, stack temporal channels, and normalize (synchronized using t_start from audio)
        video_feat, face_detected_count = preprocess_video(video_path, t_start, img_size=(64, 64)) # Shape: (15, 64, 64, 2)
        
        # Trigger modality dropout if no face is detected or if video is empty
        video_zeros = (face_detected_count == 0) or bool(np.all(video_feat == 0))
        
        if not video_zeros and video_feat.shape == (15, 64, 64, 2):
            video_feat = np.transpose(video_feat, (1, 2, 0, 3)) # Shape: (64, 64, 15, 2)
            video_feat = np.reshape(video_feat, (64, 64, 30))   # Shape: (64, 64, 30)
            if vid_mean is not None and vid_std is not None:
                video_feat = (video_feat - vid_mean.squeeze(0)) / (vid_std.squeeze(0) + 1e-6)
        else:
            video_feat = np.zeros(TARGET_VIDEO_SHAPE, dtype=np.float32)
            video_zeros = True
            
        video_feat = np.expand_dims(video_feat, axis=0) # Shape: (1, 64, 64, 30)
        
        # Predict
        motion_mean = float(np.mean(np.abs(video_feat))) if not video_zeros else 0.0
        
        # Override for completely silent and frozen states (baseline state)
        # Human breathing/balancing creates around 0.15-0.25 motion mean even when trying to be still.
        if audio_zeros and motion_mean < 0.25:
            idx = int(np.where(encoder.classes_ == "Neutral")[0][0])
            probs = np.zeros(len(encoder.classes_), dtype=float)
            probs[idx] = 1.0
            label = "Neutral"
            print(f"[DEBUG] Triggered NEUTRAL override. Mean: {motion_mean:.3f}")
        else:
            probs = model.predict({"audio_input": audio_feat, "video_input": video_feat}, verbose=0)[0]
            idx = int(np.argmax(probs))
            label = encoder.inverse_transform([idx])[0]
        
        confidences = {
            encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(probs)
        }
        
        return {
            "time": str(datetime.now()),
            "emotion": label,
            "confidence": float(probs[idx]),
            "all_scores": confidences,
            "diagnostics": {
                "face_detected_frames": face_detected_count,
                "audio_features_zero": audio_zeros,
                "video_features_zero": video_zeros
            }
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Cleanup temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)