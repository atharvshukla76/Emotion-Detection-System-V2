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

prediction_tasks = {}

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
norm_data = None
whisper_pipe = None
text_emotion_pipe = None
fer_pipe = None
mean = None
std = None
vid_mean = None
vid_std = None
# Asynchronous State Buffer for NLP
last_known_text_probs = None
last_known_transcription = ""
@app.on_event("startup")
def load_resources():
    global model, encoder, mean, std, vid_mean, vid_std, whisper_pipe, text_emotion_pipe
    try:
        model_path = os.path.join(MODEL_DIR, "multimodal_emotion_model.keras")
        encoder_path = os.path.join(MODEL_DIR, "encoder.pkl")
        norm_path = os.path.join(MODEL_DIR, "norm.pkl")
        
        print(f"[STARTUP] Loading Multi-Dataset model from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"[STARTUP WARNING] Multimodal model not found at {model_path}. You need to run training first to generate it.")
            
        else:
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

        print("Loading NLP models (Whisper + Roberta)...")
        whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
        text_emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
        
        print("Loading FER Image model...")
        fer_pipe = pipeline("image-classification", model="trpakov/vit-face-expression", top_k=None)
        print("All models loaded successfully!")
        
    except Exception as e:
        print(f"[STARTUP ERROR] Resource load failed: {e}")
        traceback.print_exc()

# =====================================================================
# 🔊 AUDIO PREPROCESSING
# =====================================================================
def preprocess_audio(file_path):
    try:
        signal, _ = librosa.load(file_path, sr=SR)
        
        # 1. VAD & Whisper Signal (Noise-reduced for NLP)
        std_sig = np.std(signal)
        whisper_signal = None
        # VAD Threshold: 0.005 (Safe for quiet webcams, but still acts as a true VAD)
        if len(signal) == 0 or std_sig < 0.005:
            print(f"[DEBUG] VAD Silence Detected (std: {std_sig:.5f}). Muting Whisper.")
            # We do NOT zero out the Emotion model! It handles its own silence gracefully.
        else:
            # Apply Noise Reduction ONLY for Whisper to remove fan/background hums.
            # prop_decrease=0.75 is safe enough to not mutilate human voices.
            whisper_signal = nr.reduce_noise(y=signal, sr=SR, prop_decrease=0.75)

        # 2. Emotion Model Signal (Strictly matching training: top_db=30, zero-center, NO scaling/NR)
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
        
        is_silent = bool(len(signal) == 0 or std_sig < 0.005)
        return np.expand_dims(features, axis=-1), t_start, whisper_signal, is_silent
    except Exception as e:
        print(f"Audio extraction error: {e}")
        return np.zeros(TARGET_AUDIO_SHAPE, dtype=np.float32), 0.0, None, True

# =====================================================================
# 🎥 VIDEO PREPROCESSING (OPTICAL FLOW)
# =====================================================================
def preprocess_video(video_path, t_start, target_frames=16, img_size=(64, 64)):
    duration = 0.0
    cap = cv2.VideoCapture(video_path)
    
    # Initialize face cascade detector
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
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
        return np.zeros(TARGET_VIDEO_SHAPE, dtype=np.float32), 0, None, None, 100.0

    # First pass: collect all detected face boxes across frames
    face_boxes = []
    for frame in raw_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(60, 60))
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
            
            stable_box = (avg_x, avg_y, avg_w, avg_h)
    else:
        stable_box = None
        face_detected_count = 0

    # Second pass: crop and stack regions
    processed_frames = []
    
    def align_to_ravdess(frame_gray, face_box):
        x, y, w_f, h_f = face_box
        cx = x + w_f / 2
        cy = y + h_f / 2
        
        target_w = int(w_f / 0.273)
        target_h = int(target_w * (720 / 1280))
        
        top_left_x = int(cx - target_w * 0.5)
        top_left_y = int(cy - target_h * 0.45)
        
        aligned = np.zeros((target_h, target_w), dtype=np.uint8)
        
        src_y1 = max(0, top_left_y)
        src_y2 = min(frame_gray.shape[0], top_left_y + target_h)
        src_x1 = max(0, top_left_x)
        src_x2 = min(frame_gray.shape[1], top_left_x + target_w)
        
        dst_y1 = max(0, -top_left_y)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        dst_x1 = max(0, -top_left_x)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        
        if src_y2 > src_y1 and src_x2 > src_x1:
            aligned[dst_y1:dst_y2, dst_x1:dst_x2] = frame_gray[src_y1:src_y2, src_x1:src_x2]
        return aligned
    
    for frame in raw_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if stable_box is not None:
            gray = align_to_ravdess(gray, stable_box)
            
        h, w = gray.shape
        
        # Use static geometric crop matching the training pipeline exactly
        # By aligning the frame first, these static coordinates perfectly slice the eyes and mouth
        # exactly like the laboratory RAVDESS dataset, regardless of webcam distance/framing!
        eyes = gray[int(h*0.15):int(h*0.45), int(w*0.2):int(w*0.8)]
        mouth = gray[int(h*0.65):int(h*0.9), int(w*0.25):int(w*0.75)]
            
        if eyes.size == 0 or mouth.size == 0:
            continue
            
        eyes_res = cv2.resize(eyes, (img_size[0], img_size[1] // 2))
        mouth_res = cv2.resize(mouth, (img_size[0], img_size[1] // 2))
        processed_frames.append(np.vstack([eyes_res, mouth_res]))
        
    if len(processed_frames) < 2:
        return np.zeros(TARGET_VIDEO_SHAPE, dtype=np.float32), 0, None, None, 100.0
        
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
            
    video_feat = np.array(flow_seq, dtype=np.float32)
    
    # Calculate overall video brightness to detect dim rooms
    vid_mean_brightness = np.mean([np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in raw_frames]) if raw_frames else 100.0

    # Return multiple frames for multi-frame FER averaging (reduces single-frame noise)
    fer_frames = []
    if len(raw_frames) > 0:
        fer_indices = [0, len(raw_frames)//2, len(raw_frames)-1]
        fer_frames = [raw_frames[i] for i in fer_indices]
        
    return video_feat, face_detected_count, fer_frames, stable_box, vid_mean_brightness

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
    import os
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
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
async def predict_emotion_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Run training and ensure multimodal_emotion_model.keras exists.")
        
    task_id = str(uuid.uuid4())
    prediction_tasks[task_id] = {"status": "processing"}
    
    # Create a secure temporary directory to isolate file processing
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "uploaded_capture.mp4")
    audio_path = os.path.join(temp_dir, "extracted_audio.wav")
    
    try:
        # Save video stream
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Start background processing
        thread = threading.Thread(target=process_prediction_task, args=(task_id, temp_dir, video_path, audio_path))
        thread.start()
        
        return {"task_id": task_id}
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    task = prediction_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
        
    if task["status"] == "failed":
        error = task.get("error", "Unknown error")
        del prediction_tasks[task_id]
        raise HTTPException(status_code=500, detail=error)
        
    if task["status"] == "completed":
        res = task["result"]
        res["status"] = "completed"
        del prediction_tasks[task_id]
        return res
        
    return {"status": "processing"}

def process_prediction_task(task_id: str, temp_dir: str, video_path: str, audio_path: str):
    try:
        # Isolate and reset the Asynchronous State Buffer for this new video
        import sys
        _buf = sys.modules[__name__]
        _buf.last_known_text_probs = None
        _buf.last_known_transcription = ""

        # Extract Audio track via ffmpeg (cross-platform)
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", audio_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # Preprocess Audio
        audio_feat, t_start, clean_signal, is_silent = preprocess_audio(audio_path)
        print(f"[DEBUG] audio_feat shape after preprocess: {audio_feat.shape}, active window start: {t_start:.2f}s")
        print(f"[DEBUG] mean shape: {np.array(mean).shape}, std shape: {np.array(std).shape}")
        
        # We define "audio_zeros" logically based on VAD silence, so the fusion engine can trigger Vision-Only mode.
        audio_zeros = is_silent
        if not bool(np.all(audio_feat == 0)):
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
        video_feat, face_detected_count, fer_frames, stable_box, vid_mean_brightness = preprocess_video(video_path, t_start, img_size=(64, 64)) # Shape: (15, 64, 64, 2)
        
        # Trigger modality dropout if no face is detected or if video is empty
        video_zeros = (face_detected_count == 0) or bool(np.all(video_feat == 0))
        
        if not video_zeros and video_feat.shape == (15, 64, 64, 2):
            video_feat = np.transpose(video_feat, (1, 2, 0, 3)) # Shape: (64, 64, 15, 2)
            video_feat = np.reshape(video_feat, (64, 64, 30))   # Shape: (64, 64, 30)
            if vid_mean is not None and vid_std is not None:
                video_feat = (video_feat - vid_mean.squeeze(0)) / (vid_std.squeeze(0) + 1e-6)
        else:
            video_feat = np.zeros((64, 64, 30), dtype=np.float32)
            video_zeros = True
            
        video_feat = np.expand_dims(video_feat, axis=0) # Shape: (1, 64, 64, 30)
        
        # Predict
        motion_mean = float(np.mean(np.abs(video_feat))) if not video_zeros else 0.0
        transcript_text = ""
        
        # Override removed: We now rely on dynamic fusion so smiles don't get forced to Neutral.
        # --- VISION-ONLY MAGNIFICATION ---
        # If the user is silent, their facial optical flow is much smaller than the speaking actors in the training data.
        # We magnify the subtle silent expressions so the network can categorize them correctly.
        if audio_zeros:
            video_feat = video_feat * 2.5
            
        probs = model.predict({"audio_input": audio_feat, "video_input": video_feat}, verbose=0)[0]
        
        # --- QUAD-MODAL FUSION (Static FER + Text NLP + Dynamic RAVDESS AV) ---
        
        # 1. Static Facial Expression Recognition (FER)
        fer_probs = np.zeros_like(probs)
        # fer_frames is a list of 1-3 frames for multi-frame averaging
        if fer_pipe is not None and fer_frames and len(fer_frames) > 0:
            try:
                label_map_fer = {
                    "happy": "Happy", "sad": "Sad", "angry": "Angry",
                    "fear": "Fear", "disgust": "Disgust", "neutral": "Neutral", "surprise": "Neutral",
                    # trpakov/vit-face-expression uses capitalized labels
                    "Happy": "Happy", "Sad": "Sad", "Angry": "Angry",
                    "Fear": "Fear", "Disgust": "Disgust", "Neutral": "Neutral", "Surprise": "Neutral"
                }
                
                # Average FER predictions across multiple frames for stability
                frame_predictions = []
                face_cascade_fer = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                
                for frame in fer_frames:
                    gray_fer = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces_fer = face_cascade_fer.detectMultiScale(gray_fer, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                    
                    if len(faces_fer) == 0:
                        continue
                        
                    # Get the largest face in this specific frame
                    x_c, y_c, w_c, h_c = max(faces_fer, key=lambda f: f[2] * f[3])
                    h_m, w_m, _ = frame.shape
                    
                    x1, y1 = max(0, x_c), max(0, y_c)
                    x2, y2 = min(w_m, x_c + w_c), min(h_m, y_c + h_c)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    crop_h, crop_w = face_crop.shape[:2]
                    
                    # Reject tiny or degenerate crops
                    if face_crop.size == 0 or crop_h < 30 or crop_w < 30:
                        continue
                    
                    # --- DIM LIGHT PROTECTOR (Brightness Boost without CLAHE) ---
                    # Safely boosts brightness if the room is completely dark, without altering local contrast
                    face_lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
                    l_chan, a_chan, b_chan = cv2.split(face_lab)
                    mean_brightness = np.mean(l_chan)
                    
                    if mean_brightness < 80:
                        boost = min(70, int(120 - mean_brightness))
                        l_chan = cv2.add(l_chan, boost)
                        face_lab = cv2.merge([l_chan, a_chan, b_chan])
                        face_crop_processed = cv2.cvtColor(face_lab, cv2.COLOR_LAB2BGR)
                    else:
                        face_crop_processed = face_crop
                    
                    face_rgb = cv2.cvtColor(face_crop_processed, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_rgb)
                    
                    fer_res = fer_pipe(pil_img)
                    single_probs = np.zeros_like(probs)
                    for res in fer_res:
                        target_label = label_map_fer.get(res['label'])
                        if target_label in encoder.classes_:
                            idx_class = int(np.where(encoder.classes_ == target_label)[0][0])
                            single_probs[idx_class] += res['score']
                    frame_predictions.append(single_probs)
                
                if frame_predictions:
                    fer_probs = np.mean(frame_predictions, axis=0)
                print(f"[DEBUG] FER Probs (avg of {len(frame_predictions)} frames): {fer_probs}")
            except Exception as e:
                print(f"[DEBUG] FER Processing failed: {e}")
        
        # 2. Text NLP Fusion (Asynchronous State Buffer)
        text_probs = np.zeros_like(probs)
        
        def run_nlp_async(signal_arr):
            global last_known_text_probs, last_known_transcription
            try:
                # whisper_pipe accepts {"raw": numpy_array, "sampling_rate": int}
                whisper_out = whisper_pipe({"raw": signal_arr, "sampling_rate": 22050})
                transcript_text = whisper_out.get("text", "").strip()
                
                # --- WHISPER HALLUCINATION FILTER ---
                # Whisper tries to transcribe silence/hum as these common phrases
                lower_text = transcript_text.lower()
                hallucinations = ["thank you", "subscribe", "thanks for watching", "by subtitlr", "subs by", "amil amara", "thank you.", "thank you!"]
                is_hallucination = any(h == lower_text.strip(" .!,") or lower_text == "" for h in hallucinations)
                
                if transcript_text and not is_hallucination:
                    print(f"[DEBUG Async] Transcription: {transcript_text}")
                    text_emotions = text_emotion_pipe(transcript_text)[0]
                    
                    temp_probs = np.zeros_like(probs)
                    label_map_text = {
                        "joy": "Happy", "sadness": "Sad", "anger": "Angry",
                        "fear": "Fear", "disgust": "Disgust", "neutral": "Neutral", "surprise": "Neutral"
                    }
                    for res in text_emotions:
                        t_label = label_map_text.get(res['label'])
                        if t_label in encoder.classes_:
                            idx_c = int(np.where(encoder.classes_ == t_label)[0][0])
                            temp_probs[idx_c] += res['score']
                    last_known_text_probs = temp_probs
                    last_known_transcription = transcript_text
            except Exception as e:
                print(f"[DEBUG Async] NLP Processing failed: {e}")

        # If speaking, launch NLP in the background so it doesn't freeze the video
        if not audio_zeros and clean_signal is not None and whisper_pipe is not None and text_emotion_pipe is not None:
            nlp_thread = threading.Thread(target=run_nlp_async, args=(clean_signal,))
            nlp_thread.start()
            nlp_thread.join(timeout=3.0)  # Wait for background transcription to complete
        
        # Pull from the state buffer instantly without waiting
        import sys
        _buf = sys.modules[__name__]
        if getattr(_buf, 'last_known_text_probs', None) is not None:
            text_probs = _buf.last_known_text_probs
            print(f"[DEBUG] Using Buffered Text Probs: {text_probs}")
        if getattr(_buf, 'last_known_transcription', None) is not None and _buf.last_known_transcription != "":
            transcript_text = _buf.last_known_transcription
        
        # 3. Ultimate Quad-Modal Fusion Weights
        # PHILOSOPHY: Physical context (Face + Tone) ALWAYS dominates. Text is only a minor hint.
        print(f"[DEBUG] Base AV Probs: {probs}")
        if audio_zeros:
            # Vision-Only / Silent Mode
            transcript_text = "[Silence]"
            has_fer = np.sum(fer_probs) > 0
            
            if has_fer:
                if motion_mean > 0.3:
                    # User is silent but moving expressively - let physical motion contribute slightly
                    probs = (probs * 0.25) + (fer_probs * 0.75)
                else:
                    # Pure silence, static - trust face completely
                    probs = fer_probs
            # If has_fer is False, 'probs' remains the base AV model (which might be noise due to silence, but it's the only fallback)
            
        else:
            # Quad-Modal Active Mode
            has_fer = np.sum(fer_probs) > 0
            has_text = np.sum(text_probs) > 0
            
            # 1. Base Confidence Extraction
            conf_vision = np.max(fer_probs) if has_fer else 0.0
            conf_tone = np.max(probs)
            conf_text = np.max(text_probs) if has_text else 0.0
            
            # 2. Extract Top Emotions
            top_vision = encoder.classes_[int(np.argmax(fer_probs))] if has_fer else None
            top_tone = encoder.classes_[int(np.argmax(probs))]
            top_text = encoder.classes_[int(np.argmax(text_probs))] if has_text else None
            
            # 3. Dynamic Alignment (Consensus) Check
            align_phys = (top_vision == top_tone) if has_fer else False           # Face matches Tone
            align_semantic_tone = (top_text == top_tone) if has_text else False   # Text matches Tone
            align_semantic_face = (top_text == top_vision) if (has_text and has_fer) else False # Text matches Face
            
            # 4. Contextual Attention Weighting (Macro vs Micro Expressions)
            # Static Vision (FER) = Macro Expressions (Exaggerated)
            # AV Model (SAMM/Optical Flow) = Tone + Micro Expressions (Subtle smirks, twitches)
            
            w_text = 0.15 if conf_text > 0.4 else 0.0
            
            if has_fer:
                if conf_vision < 0.60:
                    # MICRO-EXPRESSION DETECTED: Normal environment, but FER is uncertain because the expression is too subtle.
                    # Shift majority authority to the SAMM-trained AV model which uses optical flow.
                    print(f"[DEBUG] Micro-Expression Detected (FER conf: {conf_vision:.2f}). Shifting power to SAMM AV Model.")
                    w_tone = 0.65
                    w_vision = 0.20
                else:
                    # MACRO-EXPRESSION DETECTED: Normal environment, FER is highly confident. Balance equally.
                    print(f"[DEBUG] Macro-Expression Detected (FER conf: {conf_vision:.2f}). Balancing power.")
                    w_vision = 0.45
                    w_tone = 0.40
            else:
                w_vision = 0.0
                w_tone = 0.85
            
            is_sarcasm = False
            
            # SCENARIO A: Complete Alignment (Normal Prediction)
            if align_phys and align_semantic_tone:
                # All modalities agree. Maintain base weights.
                pass
                
            # SCENARIO B: Classic Sarcasm / Text Deception
            elif align_phys and not align_semantic_tone:
                # Face and Tone agree, but Text contradicts them (e.g., "I'm so happy" said angrily)
                print(f"[DEBUG] Contextual Consensus: Physical alignment ({top_vision}) contradicts Text ({top_text}). Triggering Sarcasm Override.")
                is_sarcasm = True
                w_text = 0.0
                w_vision = 0.60
                w_tone = 0.40
                
            # SCENARIO C: Deadpan Sarcasm
            elif not align_phys and align_semantic_face:
                # Face and Text agree (e.g. neutral face, neutral text), but Tone is highly emotional
                if conf_tone > 0.6:
                    print(f"[DEBUG] Contextual Consensus: Deadpan/Tone override. Tone ({top_tone}) contradicts Face/Text ({top_vision}).")
                    is_sarcasm = True
                    w_text = 0.0
                    w_tone = 0.60
                    w_vision = 0.40
                    
            # SCENARIO D: Total Disagreement (Complex Context like "Get the f*** out")
            elif not align_phys and not align_semantic_tone and not align_semantic_face:
                # If everything disagrees, the AI must trust the most CONFIDENT physical modality, and ignore Text.
                print(f"[DEBUG] Contextual Consensus: Total Disagreement. Trusting physical confidence.")
                w_text = 0.0
                if conf_vision > conf_tone:
                    w_vision = 0.70
                    w_tone = 0.30
                else:
                    w_vision = 0.30
                    w_tone = 0.70
            
            # Fallback for missing Face
            if not has_fer:
                w_vision = 0.0
                if w_text > 0.0:
                    w_tone = 0.75
                    w_text = 0.25
                else:
                    w_tone = 1.0
            
            # Fallback for missing Text
            if not has_text:
                w_text = 0.0
                if has_fer:
                    # Distribute based on confidence
                    if conf_vision > conf_tone:
                        w_vision = 0.60
                        w_tone = 0.40
                    else:
                        w_vision = 0.40
                        w_tone = 0.60
            
            # --- ENVIRONMENTAL SHIELD (Attention Routing) ---
            # Must run AFTER sarcasm logic so it cannot be overwritten by false sarcasm!
            # If we detect extreme brightness (dim room) or extreme optical flow (harsh outdoor shadows/grain),
            # the SAMM AV model's optical flow is completely corrupted. We shift 100% of visual power to the ViT model.
            if has_fer and (motion_mean > 0.3 or vid_mean_brightness < 80):
                print(f"[DEBUG] Harsh Environment Shield Activated! (brightness: {vid_mean_brightness:.1f}, motion: {motion_mean:.2f}). Overriding sarcasm and trusting ViT.")
                w_vision = 0.85
                w_tone = 0.15
                w_text = 0.0
                is_sarcasm = False
                        
            # Normalize Weights
            total_w = w_vision + w_tone + w_text
            if total_w > 0:
                w_vision /= total_w
                w_tone /= total_w
                w_text /= total_w
            else:
                w_tone = 1.0
                
            print(f"[DEBUG] Final Attention Weights -> Vision: {w_vision:.2f}, Tone: {w_tone:.2f}, Text: {w_text:.2f}")
            
            # 5. Apply Fusion
            probs = (probs * w_tone)
            if has_fer:
                probs += (fer_probs * w_vision)
            if has_text:
                probs += (text_probs * w_text)
            
        probs = probs / (np.sum(probs) + 1e-6) # Re-normalize
        print(f"[DEBUG] Final Fused Probs: {probs}")
        
        idx = int(np.argmax(probs))
        label = encoder.inverse_transform([idx])[0]

        confidences = {
            encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(probs)
        }
        
        prediction_tasks[task_id] = {
            "status": "completed",
            "result": {
                "time": str(datetime.now()),
                "emotion": label,
                "confidence": float(probs[idx]),
                "all_scores": confidences,
                "transcription": transcript_text,
                "diagnostics": {
                    "face_detected_frames": face_detected_count,
                    "audio_features_zero": audio_zeros,
                    "video_features_zero": video_zeros
                }
            }
        }
    except Exception as e:
        traceback.print_exc()
        prediction_tasks[task_id] = {
            "status": "failed",
            "error": f"Prediction failed: {str(e)}"
        }
    finally:
        # Cleanup temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)