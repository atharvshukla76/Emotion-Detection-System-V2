"""
MoodWave V2.0 — Quad-Modal Emotion Detection API
Upgrades applied:
  1. Six emotion classes (as per research paper): Angry, Disgust, Fear, Happy, Neutral, Sad
  2. MediaPipe Face Mesh replaces Haar cascade (works in dim light, shadows, partial occlusion)
  3. Whisper-base replaces whisper-tiny (more accurate transcription)
  4. Robust speaker isolation: Energy gate for silence masking
  5. Lip-sync diarization: ignores off-camera speakers
  6. Micro-expression detection via optical flow on tight face ROI (eyes + mouth)
  7. Sarcasm detection with visual badge flag in response
  8. 6-class DistilRoBERTa label mapping (consistent with RAVDESS + CREMA-D + SAMM)
  9. Dynamic fusion weights for all 4 modalities
  10. Bugfix: Synchronous NLP text extraction to prevent race conditions
  11. Bugfix: Audio extraction mirrors training precisely (no distribution shifts via external denoising)
  12. Bugfix: Temporal consistency for optical flow (last known bbox tracking for skipped frames)
"""

import os, cv2, pickle, shutil, tempfile, threading, subprocess, traceback, uuid
import numpy as np
import librosa
from transformers import pipeline
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from PIL import Image

# ──────────────────────────────────────────────
# 1. GLOBAL CONFIG & STATE
# ──────────────────────────────────────────────
SR          = 22050
DURATION    = 3
SAMPLES     = SR * DURATION
N_MELS      = 96
N_MFCC      = 40
N_FFT       = 2048
HOP_LENGTH  = 512
MAX_FRAMES  = 150

TARGET_AUDIO_SHAPE   = (150, 136, 1)
TARGET_VIDEO_SHAPE   = (15, 64, 64, 2)
MODEL_DIR            = "saved_model"

# 6-class emotion definitions (per research paper: RAVDESS + CREMA-D + SAMM)
POSITIVE_EMOTIONS = {"Happy"}
NEGATIVE_EMOTIONS = {"Angry", "Disgust", "Fear", "Sad"}

# HuggingFace ViT FER → our 6 classes
FER_LABEL_MAP = {
    "happy":    "Happy",  "Happy":    "Happy",
    "sad":      "Sad",    "Sad":      "Sad",
    "angry":    "Angry",  "Angry":    "Angry",
    "fear":     "Fear",   "Fear":     "Fear",
    "disgust":  "Disgust","Disgust":  "Disgust",
    "neutral":  "Neutral","Neutral":  "Neutral",
}

# DistilRoBERTa → our 6 classes
TEXT_LABEL_MAP = {
    "joy":      "Happy",
    "sadness":  "Sad",
    "anger":    "Angry",
    "fear":     "Fear",
    "disgust":  "Disgust",
    "neutral":  "Neutral",
}

prediction_tasks = {}

app = FastAPI(title="MoodWave V2.0 — Quad-Modal API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared AI Models
model             = None
encoder           = None
whisper_pipe      = None
text_emotion_pipe = None
fer_pipe          = None
mp_face_mesh      = None   

# =====================================================================
# PYTORCH QUAD-MODAL ARCHITECTURE (MOBILENETV2)
# =====================================================================
class AudioBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.channel_map = nn.Conv2d(1, 3, kernel_size=1)
        self.mobilenet.classifier = nn.Identity()
        self.dense = nn.Linear(1280, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.channel_map(x)
        x = self.mobilenet(x)
        x = self.drop(x)
        return self.relu(self.dense(x))

class VideoBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.channel_map = nn.Conv2d(30, 3, kernel_size=1)
        self.mobilenet.classifier = nn.Identity()
        self.dense = nn.Linear(1280, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        # x is [batch, 15, 64, 64, 2]
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # [batch, 15, 2, 64, 64]
        batch_size = x.size(0)
        x = x.view(batch_size, 30, 64, 64)          # [batch, 30, 64, 64]
        x = self.channel_map(x)
        x = self.mobilenet(x)
        x = self.drop(x)
        return self.relu(self.dense(x))

class SEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 32); self.fc2 = nn.Linear(32, 128)
        self.relu = nn.ReLU(); self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        sq = x.mean(dim=1)
        ex = self.sigmoid(self.fc2(self.relu(self.fc1(sq)))).unsqueeze(1)
        return x * ex

class QuadModalModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.audio = AudioBranch()
        self.video = VideoBranch()
        self.se = SEBlock()
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        self.fc1 = nn.Linear(128 * 2, 128)
        self.drop1 = nn.Dropout(0.6) 
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.3)
        self.out = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, aud, vid, modality_flags=None):
        a = self.audio(aud).unsqueeze(1)
        v = self.video(vid).unsqueeze(1)
        
        if modality_flags is not None:
            a = a * modality_flags[:, 0:1].unsqueeze(-1)
            v = v * modality_flags[:, 1:2].unsqueeze(-1)
            
        x = self.se(torch.cat([a, v], dim=1))
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x.view(x.size(0), -1)
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        return self.out(x)

# ──────────────────────────────────────────────
# 2. STARTUP — LOAD ALL MODELS
# ──────────────────────────────────────────────

@app.on_event("startup")
def load_resources():
    global model, encoder
    global whisper_pipe, text_emotion_pipe, fer_pipe, mp_face_mesh

    print("[STARTUP] Initializing Quad-Modal Engine (6-class: Angry, Disgust, Fear, Happy, Neutral, Sad)...")

    # ── AV Model ──
    try:
        model = QuadModalModel(num_classes=6)
        # BUG FIX: Use Phase 2 model as it provides equal accuracy (61.01%) but is robustly generalized.
        model_path = "best_pytorch_model_phase2.pt"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            print(f"[STARTUP] PyTorch AV model loaded: {model_path}")
        else:
            model = None
            print(f"[WARNING] PyTorch AV model {model_path} not found — prediction will use FER+Text only.")
    except Exception as e:
        model = None
        print(f"[WARNING] Failed to load PyTorch model: {e}")

    if os.path.exists(os.path.join(MODEL_DIR, "encoder.pkl")):
        with open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb") as f:
            encoder = pickle.load(f)
            
    # BUG FIX: Removed norm.pkl loading completely. The training pipeline NEVER applied z-score normalisation.
    # Applying arbitrary mean/std shifts the input distribution, breaking predictions.

    # ── Whisper BASE (more accurate than tiny) ──
    print("[STARTUP] Loading Whisper-base...")
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base.en",
        chunk_length_s=30,
    )

    # ── DistilRoBERTa text emotion (7 classes) ──
    print("[STARTUP] Loading DistilRoBERTa...")
    text_emotion_pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
    )

    # ── HuggingFace ViT FER (7 classes including Surprise) ──
    print("[STARTUP] Loading ViT FER model...")
    fer_pipe = pipeline(
        "image-classification",
        model="trpakov/vit-face-expression",
        top_k=None,
    )

    # ── MediaPipe Face Mesh (replaces Haar cascade) ──
    try:
        import mediapipe as mp
        try:
            mp_face = mp.solutions.face_mesh
        except AttributeError:
            from mediapipe.python.solutions import face_mesh as mp_face
        
        mp_face_mesh = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        print("[STARTUP] MediaPipe Face Mesh loaded.")
    except ImportError:
        print("[STARTUP] MediaPipe not installed — using Haar cascade fallback.")
        mp_face_mesh = None

    print("[STARTUP] All systems online. 6-class quad-modal engine ready.")

# ──────────────────────────────────────────────
# 3. AUDIO PREPROCESSING
# ──────────────────────────────────────────────
def preprocess_audio(file_path):
    # BUG FIX: Removed bandpass_filter and noisereduce to strictly mirror the raw processing in train_pipeline_fixed.py
    try:
        signal, _ = librosa.load(file_path, sr=SR)
        rms = float(np.sqrt(np.mean(signal ** 2)))
        is_silent = (rms < 0.002)

        whisper_signal = None
        if not is_silent:
            whisper_signal = librosa.resample(signal, orig_sr=SR, target_sr=16000).astype(np.float32)

        trimmed, trim_idx = librosa.effects.trim(signal, top_db=25) if not is_silent else (signal, [])
        t_start = float(trim_idx[0]) / SR if len(trim_idx) > 0 else 0.0
        
        sig = trimmed if len(trimmed) > 0 else signal
        sig = sig - np.mean(sig)

        if len(sig) > SAMPLES:
            start = (len(sig) - SAMPLES) // 2
            sig = sig[start:start + SAMPLES]
        else:
            pad = SAMPLES - len(sig)
            sig = np.pad(sig, (0, pad))
            
        sig = np.nan_to_num(sig).astype(np.float32)

        if np.all(sig == 0):
            return np.zeros(TARGET_AUDIO_SHAPE, dtype=np.float32), 0.0, whisper_signal, True

        mel = librosa.feature.melspectrogram(y=sig, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel)
        mfcc = librosa.feature.mfcc(y=sig, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        features = np.concatenate((mel_db, mfcc), axis=0).T
        
        if features.shape[0] > MAX_FRAMES:
            features = features[:MAX_FRAMES, :]
        else:
            pad = MAX_FRAMES - features.shape[0]
            features = np.pad(features, ((0, pad), (0, 0)))
            
        features = np.nan_to_num(features)
        features = np.clip(features, -100.0, 100.0)
        
        return np.expand_dims(features, axis=-1), t_start, whisper_signal, is_silent

    except Exception as e:
        print(f"[ERROR] Audio preprocessing: {e}")
        return np.zeros(TARGET_AUDIO_SHAPE, dtype=np.float32), 0.0, None, True


# ──────────────────────────────────────────────
# 4. VIDEO PREPROCESSING (Fixed to Match Training)
# ──────────────────────────────────────────────
def detect_face_mediapipe(frame_bgr):
    if mp_face_mesh is None: return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)
    if not results.multi_face_landmarks: return None
    lm = results.multi_face_landmarks[0].landmark
    h, w = frame_bgr.shape[:2]
    xs = [int(l.x * w) for l in lm]
    ys = [int(l.y * h) for l in lm]
    x1, x2 = max(0, min(xs) - 20), min(w, max(xs) + 20)
    y1, y2 = max(0, min(ys) - 20), min(h, max(ys) + 20)
    if (x2 - x1) < 30 or (y2 - y1) < 30: return None
    return (x1, y1, x2 - x1, y2 - y1)

def detect_face_haar(gray):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eq = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(eq, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
    if len(faces) == 0: return None
    return max(faces, key=lambda f: f[2] * f[3])

def preprocess_video(video_path, t_start=0.0):
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame  = min(int(t_start * fps), max(0, total_frames - 16))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        gray_faces   = []
        fer_frames   = []
        brightness   = []
        face_count   = 0
        last_bbox    = None

        for _ in range(16):
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            bbox = detect_face_mediapipe(frame)
            if bbox is None: bbox = detect_face_haar(gray)

            # BUG FIX: If no face detected on this specific frame, fall back to last known face 
            # to preserve structural temporal consistency for the optical flow matrices.
            if bbox is not None:
                face_count += 1
                last_bbox = bbox
            elif last_bbox is not None:
                bbox = last_bbox
            else:
                # If first frame completely fails, just use a center bounding box lock
                gh, gw = gray.shape
                bbox = (gw//4, gh//4, gw//2, gh//2)
                last_bbox = bbox

            x, y, w, h = bbox
            x, y = max(0, x), max(0, y)
            w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
            
            face_gray   = cv2.resize(gray[y:y+h, x:x+w], (64, 64))
            
            y_start_eye, y_end_eye = int(64 * 0.15), int(64 * 0.45)
            x_start_eye, x_end_eye = int(64 * 0.2), int(64 * 0.8)
            y_start_mouth, y_end_mouth = int(64 * 0.65), int(64 * 0.9)
            x_start_mouth, x_end_mouth = int(64 * 0.25), int(64 * 0.75)
            
            eyes_brow = face_gray[y_start_eye:y_end_eye, x_start_eye:x_end_eye]
            mouth = face_gray[y_start_mouth:y_end_mouth, x_start_mouth:x_end_mouth]
            
            eyes_resized = cv2.resize(eyes_brow, (64, 32))
            mouth_resized = cv2.resize(mouth, (64, 32))
            combined_roi = np.vstack([eyes_resized, mouth_resized])
            
            gray_faces.append(combined_roi)
            fer_frames.append(frame[y:y+h, x:x+w])
            brightness.append(float(np.mean(face_gray)))

        cap.release()

        if len(gray_faces) < 2:
            return np.zeros((1, *TARGET_VIDEO_SHAPE), dtype=np.float32), False, [], 0, 0.0, 100.0, 0.0

        flow_list       = []
        mouth_flow_vals = []
        
        target_frames = 15
        indices = np.linspace(0, len(gray_faces) - 1, target_frames + 1).astype(int)
        selected_frames = [gray_faces[i] for i in indices]

        for i in range(len(selected_frames) - 1):
            prev = selected_frames[i]
            nxt = selected_frames[i + 1]
            flow = cv2.calcOpticalFlowFarneback(
                prev, nxt, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            flow = np.clip(flow, -50.0, 50.0)
            flow = np.nan_to_num(flow)
            flow_list.append(flow) # Shape (64, 64, 2)

            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mouth_flow_vals.append(float(np.mean(mag[32:64, :])))

        while len(flow_list) < 15:
            flow_list.append(np.zeros((64, 64, 2), dtype=np.float32))
        flow_list = flow_list[:15]

        vid_feat = np.array(flow_list, dtype=np.float32)

        all_mags = [cv2.cartToPolar(f[..., 0], f[..., 1])[0] for f in flow_list]
        motion_mean    = float(np.mean(np.stack(all_mags)))
        mean_brightness = float(np.mean(brightness)) if brightness else 100.0
        mouth_variance = float(np.mean(mouth_flow_vals)) if mouth_flow_vals else 0.0

        return (
            np.expand_dims(vid_feat, axis=0),
            face_count > 0,
            fer_frames,
            face_count,
            motion_mean,
            mean_brightness,
            mouth_variance,
        )
    except Exception as e:
        print(f"[ERROR] Video preprocessing: {e}")
        traceback.print_exc()
        return np.zeros((1, *TARGET_VIDEO_SHAPE), dtype=np.float32), False, [], 0, 0.0, 100.0, 0.0

# ──────────────────────────────────────────────
# REST OF THE PIPELINE 
# ──────────────────────────────────────────────

WHISPER_HALLUCINATIONS = {
    "", "thank you", "thanks for watching", "please subscribe",
    "by subtitlr", "subtitles by", "thank you for watching",
    "you", ".", "...", "the", "i", "a",
}

# BUG FIX: run_nlp_async refactored to extract_text_features.
# It returns results directly instead of writing to module-level globals,
# guaranteeing thread-safety for concurrent requests.
def extract_text_features(signal_16k, n_classes):
    t_probs = np.zeros(n_classes, dtype=np.float32)
    text = ""
    try:
        out  = whisper_pipe({"raw": signal_16k, "sampling_rate": 16000})
        text = out.get("text", "").strip()
        clean = text.lower().strip(" .!?,")
        
        if not text or clean in WHISPER_HALLUCINATIONS or len(text.split()) < 3:
            return t_probs, text
            
        raw_res = text_emotion_pipe(text, top_k=None, truncation=True, max_length=512)
        while isinstance(raw_res, list) and raw_res and isinstance(raw_res[0], list):
            raw_res = raw_res[0]
        emotions = [raw_res] if isinstance(raw_res, dict) else raw_res

        for res in emotions:
            if isinstance(res, dict) and 'label' in res:
                target = TEXT_LABEL_MAP.get(res['label'])
                if target and target in encoder.classes_:
                    idx = int(np.where(encoder.classes_ == target)[0][0])
                    t_probs[idx] += float(res.get('score', 0.0))
                    
        if t_probs.sum() > 0: t_probs /= t_probs.sum()
        
    except Exception as e:
        print(f"[NLP Thread Error]: {e}")
        
    return t_probs, text

def run_fer(fer_frames, n_classes):
    probs = np.zeros(n_classes, dtype=np.float32)
    valid = 0
    for frame in fer_frames:
        try:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(rgb)
            res    = fer_pipe(pil_im, top_k=None)
            for r in res:
                target = FER_LABEL_MAP.get(r['label'])
                if target and target in encoder.classes_:
                    idx = int(np.where(encoder.classes_ == target)[0][0])
                    probs[idx] += float(r['score'])
            valid += 1
        except Exception: pass
    if valid > 0: probs /= valid
    if probs.sum() > 0: probs /= probs.sum()
    return probs

def detect_sarcasm(face_em, audio_em, text_em, face_conf, audio_conf, text_conf):
    if face_conf < 0.55 or audio_conf < 0.45: return False
    face_pos  = face_em  in POSITIVE_EMOTIONS
    audio_pos = audio_em in POSITIVE_EMOTIONS
    text_pos  = text_em  in POSITIVE_EMOTIONS if text_em else None
    if text_pos is not None and text_conf > 0.4:
        return (face_pos != audio_pos) and (face_pos != text_pos)
    return face_pos != audio_pos and face_conf > 0.65 and audio_conf > 0.55

def context_engine(probs_av, probs_fer, probs_text, has_fer, has_text, aud_silent):
    if not (has_fer and has_text and not aud_silent): return None, "partial_modalities"
    n_classes = len(encoder.classes_)
    face_idx  = int(np.argmax(probs_fer))
    audio_idx = int(np.argmax(probs_av))
    text_idx  = int(np.argmax(probs_text))
    face_em  = encoder.classes_[face_idx]
    audio_em = encoder.classes_[audio_idx]
    text_em  = encoder.classes_[text_idx]
    face_c   = float(np.max(probs_fer))
    audio_c  = float(np.max(probs_av))
    text_c   = float(np.max(probs_text))
    
    if face_em == audio_em == text_em:
        final = np.zeros(n_classes, dtype=np.float32)
        final[face_idx] = 1.0
        return final, f"all_agree:{face_em}_confirmed"
    if face_em == audio_em and face_em != text_em:
        final = (probs_fer * 0.45) + (probs_av * 0.45) + (probs_text * 0.10)
        return final, f"av+face_agree:{face_em}_boosted"
    if face_em == text_em and face_em != audio_em:
        final = (probs_fer * 0.45) + (probs_text * 0.45) + (probs_av * 0.10)
        return final, f"face+text_agree:{face_em}_boosted"
    if audio_em == text_em and audio_em != face_em:
        final = (probs_av * 0.45) + (probs_text * 0.45) + (probs_fer * 0.10)
        return final, f"av+text_agree:{audio_em}_boosted"
        
    face_pos  = face_em in POSITIVE_EMOTIONS
    audio_pos = audio_em in POSITIVE_EMOTIONS
    text_pos  = text_em in POSITIVE_EMOTIONS
    if face_pos and (not audio_pos) and (not text_pos) and face_c > 0.5 and audio_c > 0.5:
        final = (probs_av * 0.50) + (probs_text * 0.40) + (probs_fer * 0.10)
        return final, "sarcasm_detected:audio+text_win"
    if face_em != audio_em and audio_em != text_em and face_em != text_em:
        final = (probs_av * 0.50) + (probs_text * 0.30) + (probs_fer * 0.20)
        return final, "all_disagree:audio_tone_prioritized"
    return None, "no_specific_rule_hit"

def fuse_modalities(probs_av, probs_fer, probs_text, aud_silent, vid_active, motion_mean, brightness, mouth_variance):
    n = len(encoder.classes_)
    has_fer  = probs_fer.sum()  > 0.01
    has_text = probs_text.sum() > 0.01

    if not aud_silent:
        wt  = 0.25 if has_text else 0.0
        wf  = 0.25 if has_fer  else 0.0
        wav = 1.0 - wt - wf
    else:
        wt  = 0.30 if has_text else 0.0
        wf  = 0.60 if has_fer  else 0.0
        wav = max(0.10, 1.0 - wt - wf)

    if not vid_active:
        wf  = 0.0
        wav = 0.60 if not aud_silent else 0.30
        wt  = 0.40 if has_text else 0.0
        if not has_text and aud_silent: wav = 1.0
        elif not has_text: wav = 1.0

    if has_fer and motion_mean > 2.5:
        wf = 0.80; wav = 0.15; wt = 0.05 if has_text else 0.0
    if has_fer and brightness < 70:
        wf = 0.65; wav = 0.25; wt = 0.10 if has_text else 0.0
    if not has_text: wt = 0.0

    sarcasm = False
    if has_fer and not aud_silent:
        face_em  = encoder.classes_[int(np.argmax(probs_fer))]
        audio_em = encoder.classes_[int(np.argmax(probs_av))]
        text_em  = encoder.classes_[int(np.argmax(probs_text))] if has_text else None
        face_c   = float(np.max(probs_fer))
        audio_c  = float(np.max(probs_av))
        text_c   = float(np.max(probs_text)) if has_text else 0.0

        sarcasm = detect_sarcasm(face_em, audio_em, text_em, face_c, audio_c, text_c)
        if sarcasm:
            wf  = 0.10; wav = 0.50; wt  = 0.40 if has_text else 0.0

    if vid_active and not aud_silent and mouth_variance < 0.10:
        aud_silent = True
        wav = 0.0
        wf  = 0.80 if has_fer else 0.0
        wt  = 0.20 if has_text else 0.0

    total = wf + wav + wt
    if total > 0: wf /= total; wav /= total; wt /= total
    else: wav = 1.0

    final = (probs_av * wav) + (probs_fer * wf) + (probs_text * wt)
    if final.sum() > 0: final /= final.sum()

    n_idx = int(np.where(encoder.classes_ == 'Neutral')[0][0])
    if np.argmax(final) == n_idx:
        sorted_idx = np.argsort(final)[::-1]
        runner_up  = sorted_idx[1]
        if final[runner_up] > 0.25 and motion_mean > 0.3:
            em_name = encoder.classes_[runner_up]
            if not (em_name == 'Disgust' and motion_mean < 0.8):
                final[n_idx] *= 0.4
                final /= final.sum()

    weights = {"av": float(wav), "fer": float(wf), "text": float(wt)}
    return final, weights, bool(sarcasm)

def process_prediction_task(task_id, temp_dir, video_path, audio_path):
    try:
        n_classes = len(encoder.classes_)
        aud_feat, t_start, whisper_sig, aud_silent = preprocess_audio(audio_path)
        vid_feat, vid_active, fer_frames, face_count, motion_mean, brightness, mouth_var = \
            preprocess_video(video_path, t_start)

        # BUG FIX: Removed arbitrary (feat - mean) / std normalisation shift completely!

        aud_input = np.expand_dims(np.reshape(aud_feat, TARGET_AUDIO_SHAPE), axis=0)

        if model is not None:
            a_tensor = torch.from_numpy(aud_input).float()
            v_tensor = torch.from_numpy(vid_feat).float()
            
            aud_present = 0.0 if aud_silent else 1.0
            vid_present = 1.0 if vid_active else 0.0
            flags = torch.tensor([[aud_present, vid_present]], dtype=torch.float32)
            
            with torch.no_grad():
                outputs = model(a_tensor, v_tensor, flags)
                probs_av = torch.softmax(outputs, dim=1).numpy()[0]
        else:
            probs_av = np.ones(n_classes, dtype=np.float32) / n_classes

        probs_fer = np.zeros(n_classes, dtype=np.float32)
        if vid_active and fer_frames:
            probs_fer = run_fer(fer_frames, n_classes)

        probs_text = np.zeros(n_classes, dtype=np.float32)
        text_str = "[Silence]"
        
        # BUG FIX: Run NLP synchronously per-request to avoid race conditions and stale globals
        if not aud_silent and whisper_sig is not None:
            probs_text, text_str = extract_text_features(whisper_sig, n_classes)

        has_fer  = probs_fer.sum()  > 0.01
        has_text = probs_text.sum() > 0.01

        context_probs, context_note = context_engine(probs_av, probs_fer, probs_text, has_fer, has_text, aud_silent)
        final_probs, weights, sarcasm = fuse_modalities(probs_av, probs_fer, probs_text, aud_silent, vid_active, motion_mean, brightness, mouth_var)

        if context_probs is not None:
            final_probs = (context_probs * 0.70) + (final_probs * 0.30)
            if final_probs.sum() > 0: final_probs /= final_probs.sum()
        else:
            context_note = "partial_modalities"

        final_idx   = int(np.argmax(final_probs))
        pred_label  = encoder.inverse_transform([final_idx])[0]

        prediction_tasks[task_id] = {
            "status": "completed",
            "result": {
                "time":       str(datetime.now()),
                "emotion":    pred_label,
                "confidence": float(final_probs[final_idx]),
                "sarcasm":    sarcasm,
                "all_scores": {
                    encoder.inverse_transform([i])[0]: float(p)
                    for i, p in enumerate(final_probs)
                },
                "transcription": text_str,
                "fusion_weights": weights,
                "diagnostics": {
                    "face_detected_frames": face_count,
                    "audio_silent":         aud_silent,
                    "video_active":         vid_active,
                    "motion_mean":          round(motion_mean, 3),
                    "brightness":           round(brightness, 1),
                    "mouth_variance":       round(mouth_var, 3),
                    "context_engine":       context_note,
                },
            },
        }

    except Exception as e:
        print(f"[FATAL] Prediction thread: {e}")
        traceback.print_exc()
        prediction_tasks[task_id] = {"status": "failed", "error": str(e)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("moodwave.html"):
        with open("moodwave.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return "<h1>MoodWave Quad-Modal API — Online (GUI not found)</h1>"

@app.get("/health")
def health_check():
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "emotions":     list(encoder.classes_) if encoder else [],
        "n_classes":    len(encoder.classes_) if encoder else 0,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if encoder is None: raise HTTPException(status_code=500, detail="Encoder not loaded.")
    task_id  = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    ext      = os.path.splitext(file.filename or "clip.webm")[1] or ".webm"
    vid_path = os.path.join(temp_dir, f"input{ext}")
    aud_path = os.path.join(temp_dir, "audio.wav")
    with open(vid_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", vid_path, "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", aud_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
    except Exception:
        open(aud_path, "a").close()
    prediction_tasks[task_id] = {"status": "processing"}
    threading.Thread(target=process_prediction_task, args=(task_id, temp_dir, vid_path, aud_path), daemon=True).start()
    return {"task_id": task_id}

@app.get("/result/{task_id}")
def get_result(task_id: str):
    if task_id not in prediction_tasks:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    res = prediction_tasks[task_id]
    if res["status"] in ("completed", "failed"):
        del prediction_tasks[task_id]
    return res
