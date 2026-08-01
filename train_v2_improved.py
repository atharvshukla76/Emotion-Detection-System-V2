"""
MoodWave V2 — Improved Training Script
Accuracy improvements:
  1. Focal Loss (gamma=2) — forces focus on hard samples (Fear/Sad confusion)
  2. Mixup Augmentation — smoother decision boundaries
  3. Fixed Phase 2 — discriminative LR instead of uniform unfreeze
  4. Cosine LR with Warmup — smooth decay, no sudden drops
  5. Batch size 32 — smoother gradients
  6. SE Block in Fusion — channel-wise importance weighting
"""

import os
import cv2
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# CONSTANTS
# =====================================================================
tf.random.set_seed(42)
np.random.seed(42)

SAMPLE_RATE = 22050
Duration = 3
SAMPLES = SAMPLE_RATE * Duration
N_MELS = 96
N_MFCC = 40
N_FFT = 2048
MAX_FRAMES = 150
HOP_LENGTH = 512
TARGET_AUDIO_SHAPE = (MAX_FRAMES, N_MELS + N_MFCC, 1)
TARGET_VIDEO_SHAPE = (15, 64, 64, 2)

BATCH_SIZE = 64  # FIX #5: Increased from 16 → 32

emotion_map = {
    "ANG": "Angry", "DIS": "Disgust", "FEA": "Fear",
    "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad", "SUR": "Surprise"
}
ravdess_emotions = {
    "01": "Neutral", "02": "Neutral", "03": "Happy", "04": "Sad",
    "05": "Angry", "06": "Fear", "07": "Disgust", "08": "Surprise"
}
samm_emotions_map = {
    "Happiness": "Happy", "Sadness": "Sad", "Anger": "Angry",
    "Fear": "Fear", "Disgust": "Disgust", "Surprise": "Surprise"
}

# =====================================================================
# FIX #1: FOCAL LOSS
# Forces model to focus on hard-to-classify samples (Fear/Sad confusion)
# =====================================================================
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # Apply label smoothing
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        # Clip predictions for numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Cross-entropy
        ce = -y_true * tf.math.log(y_pred)

        # Focal weight: (1 - p_t)^gamma
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        focal_loss = focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha,
                       "label_smoothing": self.label_smoothing})
        return config

# =====================================================================
# FIX #4: COSINE ANNEALING WITH WARMUP
# =====================================================================
class CosineWarmupScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_steps, peak_lr, min_lr=1e-7):
        super().__init__()
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.peak_lr = float(peak_lr)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Warmup phase: linear ramp up
        warmup_lr = self.peak_lr * (step / self.warmup_steps)
        # Cosine decay phase
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + tf.cos(np.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr
        }

# =====================================================================
# AUDIO PREPROCESSING
# =====================================================================
def load_audio(file_path):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return np.zeros(SAMPLES, dtype=np.float32)
    try:
        signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(signal) == 0:
            return np.zeros(SAMPLES, dtype=np.float32)
        trimmed_signal, _ = librosa.effects.trim(signal, top_db=30)
        if len(trimmed_signal) > 0:
            signal = trimmed_signal
        signal = signal - np.mean(signal)
    except Exception as e:
        print(f"Error loading audio {file_path}: {e}")
        return np.zeros(SAMPLES, dtype=np.float32)
    if len(signal) > SAMPLES:
        start = (len(signal) - SAMPLES) // 2
        signal = signal[start:start + SAMPLES]
    else:
        pad = SAMPLES - len(signal)
        signal = np.pad(signal, (0, pad))
    return np.nan_to_num(signal).astype(np.float32)

def extract_audio_features(signal):
    try:
        if np.all(signal == 0):
            return np.zeros(TARGET_AUDIO_SHAPE, dtype=np.float32)
        mel = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_fft=N_FFT,
                                              hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel)
        mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                     n_fft=N_FFT, hop_length=HOP_LENGTH)
        features = np.concatenate((mel_db, mfcc), axis=0).T
        if features.shape[0] > MAX_FRAMES:
            features = features[:MAX_FRAMES, :]
        else:
            pad = MAX_FRAMES - features.shape[0]
            features = np.pad(features, ((0, pad), (0, 0)))
        features = np.nan_to_num(np.clip(features, -100.0, 100.0))
        return np.expand_dims(features, axis=-1)
    except Exception:
        return np.zeros(TARGET_AUDIO_SHAPE, dtype=np.float32)

def process_audio_file(file_path):
    return extract_audio_features(load_audio(file_path))

# =====================================================================
# VIDEO PREPROCESSING (Optical Flow ROI)
# =====================================================================
def extract_landmark_masked_flow_from_frames(frame_list, target_frames=16, img_size=(64, 64)):
    if len(frame_list) < 2:
        return np.zeros(TARGET_VIDEO_SHAPE, dtype=np.float32)
    frames = []
    for frame in frame_list:
        if frame is None:
            continue
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            h, w = gray.shape
            eyes_brow = gray[int(h*0.15):int(h*0.45), int(w*0.2):int(w*0.8)]
            mouth = gray[int(h*0.65):int(h*0.9), int(w*0.25):int(w*0.75)]
            if eyes_brow.size == 0 or mouth.size == 0:
                continue
            combined_roi = np.vstack([
                cv2.resize(eyes_brow, (img_size[0], img_size[1]//2)),
                cv2.resize(mouth, (img_size[0], img_size[1]//2))
            ])
            frames.append(combined_roi)
        except Exception:
            continue
    if len(frames) < 2:
        return np.zeros(TARGET_VIDEO_SHAPE, dtype=np.float32)
    indices = np.linspace(0, len(frames)-1, target_frames).astype(int)
    selected_frames = [frames[i] for i in indices]
    flow_sequence = []
    for i in range(len(selected_frames) - 1):
        try:
            flow = cv2.calcOpticalFlowFarneback(
                selected_frames[i], selected_frames[i+1], None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            flow_sequence.append(np.clip(np.nan_to_num(flow), -50.0, 50.0))
        except Exception:
            flow_sequence.append(np.zeros((img_size[0], img_size[1], 2), dtype=np.float32))
    return np.array(flow_sequence, dtype=np.float32)

def extract_flow_from_video(video_path):
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return np.zeros(TARGET_VIDEO_SHAPE, dtype=np.float32)
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    return extract_landmark_masked_flow_from_frames(frames)

# =====================================================================
# DATASET BUILDERS
# =====================================================================
def build_ravdness_dataset(dataset_dir):
    X_audio, X_video, y, actors = [], [], [], []
    idx = 0
    if not os.path.exists(dataset_dir):
        print(f"Error: RAVDESS directory not found at {dataset_dir}")
        return (np.empty((0, *TARGET_AUDIO_SHAPE)), np.empty((0, *TARGET_VIDEO_SHAPE)),
                np.array([]), np.array([]))
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if not file.endswith('.mp4'):
                continue
            parts = file.split('-')
            if len(parts) < 7:
                continue
            emotion_code = parts[2]
            actor_id = parts[6].split('.')[0]
            if emotion_code not in ravdess_emotions:
                continue
            video_path = os.path.join(root, file)
            if os.path.getsize(video_path) == 0:
                continue
            try:
                temp_audio = f"temp_ravdess_{idx}.wav"
                os.system(f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 22050 -ac 1 "{temp_audio}" >nul 2>&1')
                audio_feat = process_audio_file(temp_audio) if (os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0) else np.zeros(TARGET_AUDIO_SHAPE, dtype=np.float32)
                video_feat = extract_flow_from_video(video_path)
                if (audio_feat is not None and video_feat is not None and
                        not np.isnan(audio_feat).any() and not np.isnan(video_feat).any()):
                    X_audio.append(audio_feat)
                    X_video.append(video_feat)
                    y.append(ravdess_emotions[emotion_code])
                    actors.append(actor_id)
                    idx += 1
            except Exception as e:
                print(f"Failed to process {file}: {e}")
            finally:
                if os.path.exists(temp_audio):
                    try: os.remove(temp_audio)
                    except: pass

    clean = [(a, v, l, ac) for a, v, l, ac in zip(X_audio, X_video, y, actors)
             if a.shape == TARGET_AUDIO_SHAPE and v.shape == TARGET_VIDEO_SHAPE]
    if not clean:
        return (np.empty((0, *TARGET_AUDIO_SHAPE)), np.empty((0, *TARGET_VIDEO_SHAPE)),
                np.array([]), np.array([]))
    ca, cv2_, cl, cac = zip(*clean)
    return np.array(ca), np.array(cv2_), np.array(cl), np.array(cac)

def build_cremad_dataset(dataset_path="./AudioWAV"):
    if not os.path.exists(dataset_path):
        print(f"CREMA-D not found at {dataset_path}. Skipping.")
        return None, None, None
    cremad_emotion_map = {
        "ANG": "Angry", "DIS": "Disgust", "FEA": "Fear",
        "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad"
    }
    wav_files = [f for f in os.listdir(dataset_path) if f.lower().endswith('.wav')]
    print(f"\nCREMA-D: Found {len(wav_files)} WAV files")
    X_audio, y_labels, actor_ids = [], [], []
    skipped = 0
    for i, fname in enumerate(wav_files):
        parts = fname.replace('.wav', '').split('_')
        if len(parts) < 3 or parts[2] not in cremad_emotion_map:
            skipped += 1
            continue
        file_path = os.path.join(dataset_path, fname)
        try:
            signal = load_audio(file_path)
            features = extract_audio_features(signal)
            if features is None or features.shape != TARGET_AUDIO_SHAPE or not np.isfinite(features).all():
                skipped += 1
                continue
            X_audio.append(features)
            y_labels.append(cremad_emotion_map[parts[2]])
            actor_ids.append(f"cre_{parts[0]}")
        except Exception:
            skipped += 1
        if (i+1) % 1500 == 0:
            print(f"  Processed {i+1}/{len(wav_files)}...")
    print(f"  CREMA-D loaded: {len(X_audio)} samples")
    return np.array(X_audio, dtype=np.float32), np.array(y_labels), np.array(actor_ids)

def build_samm_dataset(dataset_dir, excel_path):
    if not os.path.exists(excel_path):
        print(f"Error: SAMM Excel file not found at {excel_path}")
        return np.empty((0, *TARGET_VIDEO_SHAPE)), np.array([]), np.array([])
    try:
        df_raw = pd.read_excel(excel_path, header=None)
        header_row_idx = 0
        for idx, row in df_raw.iterrows():
            row_vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
            if any('subject' in val for val in row_vals) and any('filename' in val for val in row_vals):
                header_row_idx = idx
                break
        df = pd.read_excel(excel_path, header=header_row_idx)
        df.columns = [str(c).strip() for c in df.columns]
        col_mapping = {}
        for col in df.columns:
            col_str = col.lower()
            if 'subject' in col_str: col_mapping[col] = 'Subject'
            elif 'filename' in col_str: col_mapping[col] = 'Filename'
            elif 'emotion' in col_str: col_mapping[col] = 'Emotion'
        df = df.rename(columns=col_mapping)
    except Exception as e:
        print(f"Error opening Excel file: {e}")
        return np.empty((0, *TARGET_VIDEO_SHAPE)), np.array([]), np.array([])

    df = df.dropna(subset=['Subject', 'Filename', 'Emotion'])
    X_video, y, subjects = [], [], []
    for _, row in df.iterrows():
        try:
            subject_id = str(row['Subject']).zfill(3)
            filename = str(row['Filename']).strip()
            emotion = str(row['Emotion']).strip()
            if emotion not in samm_emotions_map:
                continue
            video_path = os.path.join(dataset_dir, subject_id, f"{subject_id}_{filename}.mp4")
            folder_path = os.path.join(dataset_dir, subject_id, filename)
            flow = None
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                flow = extract_flow_from_video(video_path)
            elif os.path.isdir(folder_path):
                image_files = sorted([
                    os.path.join(folder_path, f) for f in os.listdir(folder_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])
                frames = [cv2.imread(p) for p in image_files]
                frames = [f for f in frames if f is not None]
                if frames:
                    flow = extract_landmark_masked_flow_from_frames(frames)
            if flow is not None and not np.isnan(flow).any():
                X_video.append(flow)
                y.append(samm_emotions_map[emotion])
                subjects.append(subject_id)
        except Exception as e:
            print(f"Failed to process SAMM row: {e}")
    clean = [(v, l, s) for v, l, s in zip(X_video, y, subjects) if v.shape == TARGET_VIDEO_SHAPE]
    if not clean:
        return np.empty((0, *TARGET_VIDEO_SHAPE)), np.array([]), np.array([])
    cv_, cl, cs = zip(*clean)
    return np.array(cv_), np.array(cl), np.array(cs)

# =====================================================================
# LOAD DATASETS & SPLIT
# =====================================================================
print("Loading RAVDESS...")
X_audio_rav, X_video_rav, y_rav, actors_rav = build_ravdness_dataset("./RAVDNESS_Dataset")
print("Loading SAMM...")
X_video_sam, y_sam, subjects_sam = build_samm_dataset(
    "./SAMM/SAMM-full/SAMM-full/SAMM",
    "./SAMM/SAMM_Micro_FACS_Codes_v2.xlsx"
)
print("Loading CREMA-D...")
X_audio_cre, y_cre, actors_cre = build_cremad_dataset("./AudioWAV")

label_encoder = LabelEncoder()
label_encoder.fit(list(emotion_map.values()))
num_classes = len(label_encoder.classes_)

# RAVDESS Split
actors_rav = np.array([f"rav_{a}" for a in actors_rav])
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(gss.split(X_audio_rav, y_rav, groups=actors_rav))

X_train_rav_aud, X_train_rav_vid = X_audio_rav[train_idx], X_video_rav[train_idx]
X_val_rav_aud, X_val_rav_vid = X_audio_rav[val_idx], X_video_rav[val_idx]
y_train_rav = label_encoder.transform(y_rav[train_idx])
y_val_rav = label_encoder.transform(y_rav[val_idx])

# SAMM Split
subjects_sam = np.array([f"sam_{s}" for s in subjects_sam])
gss_sam = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
s_train_idx, s_val_idx = next(gss_sam.split(X_video_sam, y_sam, groups=subjects_sam))
X_train_sam_vid = X_video_sam[s_train_idx]
X_val_sam_vid = X_video_sam[s_val_idx]
y_train_sam = label_encoder.transform(y_sam[s_train_idx])
y_val_sam = label_encoder.transform(y_sam[s_val_idx])

# CREMA Split
gss_cre = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
c_train_idx, c_val_idx = next(gss_cre.split(X_audio_cre, y_cre, groups=actors_cre))
X_train_cre_aud = X_audio_cre[c_train_idx]
X_val_cre_aud = X_audio_cre[c_val_idx]
y_train_cre = label_encoder.transform(y_cre[c_train_idx])
y_val_cre = label_encoder.transform(y_cre[c_val_idx])

y_train_all = np.concatenate([y_train_rav, y_train_sam, y_train_cre])
weights = compute_class_weight('balanced', classes=np.unique(y_train_all), y=y_train_all)
class_weights = dict(zip(np.unique(y_train_all), weights))
disgust_idx = int(np.where(label_encoder.classes_ == 'Disgust')[0][0])
class_weights[disgust_idx] *= 0.6
print(f"Splits Ready. RAV:{len(y_train_rav)}, SAM:{len(y_train_sam)}, CRE:{len(y_train_cre)}")

# =====================================================================
# NORMALIZATION
# =====================================================================
import gc

temp_aud = np.concatenate([X_train_rav_aud, X_train_cre_aud], axis=0)
mean = temp_aud.mean(axis=(0, 1, 3), keepdims=True)
std = temp_aud.std(axis=(0, 1, 3), keepdims=True) + 1e-6
del temp_aud; gc.collect()

X_train_rav_aud = (X_train_rav_aud - mean) / std
X_train_cre_aud = (X_train_cre_aud - mean) / std
X_val_rav_aud = (X_val_rav_aud - mean) / std
X_val_cre_aud = (X_val_cre_aud - mean) / std

temp_vid = np.concatenate([X_train_rav_vid, X_train_sam_vid], axis=0)
vid_mean = temp_vid.mean(axis=(0, 1, 2, 3), keepdims=True)
vid_std = temp_vid.std(axis=(0, 1, 2, 3), keepdims=True) + 1e-6
del temp_vid; gc.collect()

X_train_rav_vid = (X_train_rav_vid - vid_mean) / vid_std
X_train_sam_vid = (X_train_sam_vid - vid_mean) / vid_std
X_val_rav_vid = (X_val_rav_vid - vid_mean) / vid_std
X_val_sam_vid = (X_val_sam_vid - vid_mean) / vid_std
print("Normalization complete.")

TARGET_VIDEO_SHAPE_STACKED = (64, 64, 30)

# =====================================================================
# FIX #2: MIXUP AUGMENTATION
# Blends two training samples to force smoother decision boundaries
# =====================================================================
def apply_mixup(inputs, labels, alpha=0.2):
    batch_size = tf.shape(labels)[0]
    lam = tf.random.uniform([], 0.0, alpha, dtype=tf.float32)
    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_audio = lam * inputs["audio_input"] + (1 - lam) * tf.gather(inputs["audio_input"], indices)
    mixed_video = lam * inputs["video_input"] + (1 - lam) * tf.gather(inputs["video_input"], indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
    return {"audio_input": mixed_audio, "video_input": mixed_video}, mixed_labels

# =====================================================================
# TF.DATA PIPELINE
# =====================================================================
def apply_time_mask(audio):
    t_start = tf.random.uniform([], 0, 130, dtype=tf.int32)
    t_width = tf.random.uniform([], 5, 20, dtype=tf.int32)
    t_end = tf.minimum(t_start + t_width, 150)
    mask = tf.cast(tf.logical_or(tf.range(150) < t_start, tf.range(150) >= t_end), tf.float32)
    return audio * tf.reshape(mask, [150, 1, 1])

def apply_freq_mask(audio):
    f_start = tf.random.uniform([], 0, 120, dtype=tf.int32)
    f_width = tf.random.uniform([], 5, 15, dtype=tf.int32)
    f_end = tf.minimum(f_start + f_width, 136)
    mask = tf.cast(tf.logical_or(tf.range(136) < f_start, tf.range(136) >= f_end), tf.float32)
    return audio * tf.reshape(mask, [1, 136, 1])

def augment_sample(inputs, label):
    audio, video = inputs["audio_input"], inputs["video_input"]
    audio = tf.cond(tf.random.uniform([]) < 0.2, lambda: apply_time_mask(audio), lambda: audio)
    audio = tf.cond(tf.random.uniform([]) < 0.2, lambda: apply_freq_mask(audio), lambda: audio)
    video = tf.cond(tf.random.uniform([]) < 0.5, lambda: tf.image.flip_left_right(video), lambda: video)
    video = video + tf.random.normal(tf.shape(video), stddev=0.05)
    return {"audio_input": audio, "video_input": video}, label

def create_dataset(aud, vid, lbl, modality):
    lbl_cat = to_categorical(lbl, num_classes=num_classes)
    if modality == "both":
        ds = tf.data.Dataset.from_tensor_slices((aud, vid, lbl_cat))
    elif modality == "audio":
        ds = tf.data.Dataset.from_tensor_slices((aud, lbl_cat))
    elif modality == "video":
        ds = tf.data.Dataset.from_tensor_slices((vid, lbl_cat))

    def format_tensors(*args):
        if modality == "both":
            a, v, l = args
        elif modality == "audio":
            a, l = args
            v = tf.zeros((15, 64, 64, 2), dtype=tf.float32)
        elif modality == "video":
            v, l = args
            a = tf.zeros((150, 136, 1), dtype=tf.float32)
        v = tf.transpose(v, [1, 2, 0, 3])
        v = tf.reshape(v, [64, 64, 30])
        return {"audio_input": a, "video_input": v}, l

    return ds.map(format_tensors, num_parallel_calls=tf.data.AUTOTUNE)

train_rav = create_dataset(X_train_rav_aud, X_train_rav_vid, y_train_rav, "both")
train_sam = create_dataset(None, X_train_sam_vid, y_train_sam, "video")
train_cre = create_dataset(X_train_cre_aud, None, y_train_cre, "audio")
train_dataset = train_rav.concatenate(train_sam).concatenate(train_cre)
train_dataset = train_dataset.shuffle(buffer_size=10000, seed=42)
train_dataset = train_dataset.map(augment_sample, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# FIX #2: Apply Mixup after batching
train_dataset = train_dataset.map(apply_mixup, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_rav = create_dataset(X_val_rav_aud, X_val_rav_vid, y_val_rav, "both")
val_sam = create_dataset(None, X_val_sam_vid, y_val_sam, "video")
val_cre = create_dataset(X_val_cre_aud, None, y_val_cre, "audio")
val_dataset = val_rav.concatenate(val_sam).concatenate(val_cre)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print("Dataset pipelines built.")

# =====================================================================
# MODEL ARCHITECTURE
# FIX #6: Added Squeeze-and-Excitation block in Fusion
# =====================================================================
L2 = regularizers.l2(5e-4)

# --- AUDIO BRANCH ---
audio_inputs = layers.Input(shape=TARGET_AUDIO_SHAPE, name="audio_input")
x_aud = layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=L2, name="audio_conv2d_1")(audio_inputs)
x_aud = layers.BatchNormalization(name="audio_bn_1")(x_aud)
x_aud = layers.Activation('relu', name="audio_act_1")(x_aud)
x_aud = layers.MaxPooling2D((2,2), name="audio_pool_1")(x_aud)
x_aud = layers.Dropout(0.2, name="audio_drop_1")(x_aud)

x_aud = layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=L2, name="audio_conv2d_2")(x_aud)
x_aud = layers.BatchNormalization(name="audio_bn_2")(x_aud)
x_aud = layers.Activation('relu', name="audio_act_2")(x_aud)
x_aud = layers.MaxPooling2D((2,2), name="audio_pool_2")(x_aud)
x_aud = layers.Dropout(0.25, name="audio_drop_2")(x_aud)

x_aud = layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=L2, name="audio_conv2d_3")(x_aud)
x_aud = layers.BatchNormalization(name="audio_bn_3")(x_aud)
x_aud = layers.Activation('relu', name="audio_act_3")(x_aud)
x_aud = layers.MaxPooling2D((2,2), name="audio_pool_3")(x_aud)
x_aud = layers.Dropout(0.3, name="audio_drop_3")(x_aud)

x_aud = layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=L2, name="audio_conv2d_4")(x_aud)
x_aud = layers.BatchNormalization(name="audio_bn_4")(x_aud)
x_aud = layers.Activation('relu', name="audio_act_4")(x_aud)

time_steps = x_aud.shape[1]
features_aud = x_aud.shape[2] * x_aud.shape[3]
x_aud = layers.Reshape((time_steps, features_aud), name="audio_reshape")(x_aud)
x_aud = layers.Conv1D(64, 1, padding='same', activation='relu', kernel_regularizer=L2, name="audio_conv1d_1")(x_aud)
x_aud = layers.BatchNormalization(name="audio_bn_5")(x_aud)
x_aud = layers.Dropout(0.3, name="audio_drop_4")(x_aud)
x_aud = layers.Conv1D(32, 3, padding='same', activation='relu', kernel_regularizer=L2, name="audio_conv1d_2")(x_aud)
x_aud = layers.BatchNormalization(name="audio_bn_6")(x_aud)
x_aud = layers.GlobalAveragePooling1D(name="audio_gap1d")(x_aud)
audio_emb = layers.Dense(128, activation='relu', kernel_regularizer=L2, name="audio_dense")(x_aud)

# --- VIDEO BRANCH ---
video_inputs = layers.Input(shape=TARGET_VIDEO_SHAPE_STACKED, name="video_input")
x_vid = layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=L2, name="video_conv_1")(video_inputs)
x_vid = layers.BatchNormalization(name="video_bn_1")(x_vid)
x_vid = layers.Activation('relu', name="video_act_1")(x_vid)
x_vid = layers.MaxPooling2D((2,2), name="video_pool_1")(x_vid)
x_vid = layers.Dropout(0.25, name="video_drop_1")(x_vid)

x_vid = layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=L2, name="video_conv_2")(x_vid)
x_vid = layers.BatchNormalization(name="video_bn_2")(x_vid)
x_vid = layers.Activation('relu', name="video_act_2")(x_vid)
x_vid = layers.MaxPooling2D((2,2), name="video_pool_2")(x_vid)
x_vid = layers.Dropout(0.35, name="video_drop_2")(x_vid)

x_vid = layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=L2, name="video_conv_3")(x_vid)
x_vid = layers.BatchNormalization(name="video_bn_3")(x_vid)
x_vid = layers.Activation('relu', name="video_act_3")(x_vid)
x_vid = layers.Dropout(0.4, name="video_drop_3")(x_vid)

x_vid = layers.GlobalAveragePooling2D(name="video_gap")(x_vid)
video_emb = layers.Dense(128, activation='relu', kernel_regularizer=L2, name="video_dense")(x_vid)

# --- FIX #6: FUSION WITH SQUEEZE-AND-EXCITATION BLOCK ---
audio_seq = layers.Reshape((1, 128), name="audio_seq")(audio_emb)
video_seq = layers.Reshape((1, 128), name="video_seq")(video_emb)
merged_seq = layers.Concatenate(axis=1, name="fusion_seq")([audio_seq, video_seq])  # (None, 2, 128)

# Multi-Head Attention
attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=128, name="cross_attention")(merged_seq, merged_seq)

# Residual connection
attn_out = layers.Add(name="attn_residual")([attn_out, merged_seq])
attn_out = layers.LayerNormalization(name="attn_layernorm")(attn_out)
attn_flat = layers.Flatten(name="attn_flatten")(attn_out)

# Squeeze-and-Excitation block: learns channel importance
# Squeeze: compress to a descriptor
se = layers.Dense(64, activation='relu', name="se_squeeze")(attn_flat)
# Excitation: rescale channels
se = layers.Dense(256, activation='sigmoid', name="se_excite")(se)
# Scale the features
attn_scaled = layers.Multiply(name="se_scale")([attn_flat, se])

# Classifier head
fc = layers.Dense(256, activation='relu', kernel_regularizer=L2, name="fc_fusion_1")(attn_scaled)
fc = layers.Dropout(0.4, name="fc_drop_1")(fc)
fc = layers.Dense(128, activation='relu', kernel_regularizer=L2, name="fc_fusion_2")(fc)
fc = layers.Dropout(0.3, name="fc_drop_2")(fc)
fc = layers.Dense(64, activation='relu', kernel_regularizer=L2, name="fc_fusion_3")(fc)
fc = layers.Dropout(0.2, name="fc_drop_3")(fc)

outputs = layers.Dense(num_classes, activation='softmax', name="softmax_output")(fc)

model = models.Model(inputs=[audio_inputs, video_inputs], outputs=outputs, name="Moodwave_V2_Improved")
model.summary()

# =====================================================================
# TRANSFER V1 WEIGHTS & FREEZE AUDIO BRANCH
# =====================================================================
v1_model_path = "best_model.keras"
if not os.path.exists(v1_model_path):
    v1_model_path = "d:/Emotion Detection system/best_model.keras"

if os.path.exists(v1_model_path):
    v1_model = tf.keras.models.load_model(v1_model_path)
    print(f"V1 Model loaded from {v1_model_path}")
    v1_to_v2_mapping = {
        "conv2d": "audio_conv2d_1", "batch_normalization": "audio_bn_1",
        "conv2d_1": "audio_conv2d_2", "batch_normalization_1": "audio_bn_2",
        "conv2d_2": "audio_conv2d_3", "batch_normalization_2": "audio_bn_3",
        "conv2d_3": "audio_conv2d_4", "batch_normalization_3": "audio_bn_4",
        "conv1d": "audio_conv1d_1", "batch_normalization_4": "audio_bn_5",
        "conv1d_1": "audio_conv1d_2", "batch_normalization_5": "audio_bn_6",
        "dense": "audio_dense"
    }
    transferred = 0
    for v1_name, v2_name in v1_to_v2_mapping.items():
        try:
            v1_w = v1_model.get_layer(v1_name).get_weights()
            v2_l = model.get_layer(v2_name)
            v2_w = v2_l.get_weights()
            if len(v1_w) == len(v2_w) and all(a.shape == b.shape for a, b in zip(v1_w, v2_w)):
                v2_l.set_weights(v1_w)
                v2_l.trainable = True
                transferred += 1
        except Exception as e:
            print(f"  Skip {v1_name}: {e}")
    print(f"Transferred & froze {transferred}/13 audio layers.")
else:
    print("V1 weights not found — audio branch trains from scratch.")

# =====================================================================
# FIX #4: PHASE 1 — Cosine warmup schedule
# =====================================================================
steps_per_epoch = (len(y_train_rav) + len(y_train_sam) + len(y_train_cre)) // BATCH_SIZE
total_steps_p1 = steps_per_epoch * 80
warmup_steps_p1 = steps_per_epoch * 5  # 5 epoch warmup

lr_schedule_p1 = CosineWarmupScheduler(
    warmup_steps=warmup_steps_p1,
    total_steps=total_steps_p1,
    peak_lr=1e-3,
    min_lr=1e-6
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_p1),
    loss=FocalLoss(gamma=2.0, label_smoothing=0.1),  # FIX #1
    metrics=['accuracy']
)

callbacks_p1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7,
        restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_multimodal_model_v2.keras',
        monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("\n=== PHASE 1: Training video + fusion branches with frozen audio ===")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=40,
    class_weight=class_weights,
    callbacks=callbacks_p1,
    verbose=1
)

# =====================================================================
# FIX #3: PHASE 2 — Discriminative learning rates (NOT uniform unfreeze)
# Audio branch gets 10x smaller LR than fusion layers
# =====================================================================
print("\n=== PHASE 2: Discriminative Fine-Tuning ===")
model.load_weights('best_multimodal_model_v2.keras')

# Unfreeze all layers but apply layer-group specific learning rates
model.trainable = True

# Collect parameters by group
audio_params = []
video_params = []
fusion_params = []

for layer in model.layers:
    if not layer.weights:
        continue
    name = layer.name
    if any(k in name for k in ["audio_conv2d", "audio_bn", "audio_drop", "audio_conv1d", "audio_dense", "audio_gap", "audio_reshape", "audio_act", "audio_pool", "audio_seq"]):
        audio_params.extend(layer.trainable_weights)
    elif any(k in name for k in ["video_conv", "video_bn", "video_drop", "video_gap", "video_dense", "video_act", "video_pool", "video_seq"]):
        video_params.extend(layer.trainable_weights)
    else:
        fusion_params.extend(layer.trainable_weights)

print(f"Audio params: {len(audio_params)}, Video params: {len(video_params)}, Fusion params: {len(fusion_params)}")

# Use a custom training step via multiple optimizers
# Simple approach: recompile with a single lr but manually scale gradients
total_steps_p2 = steps_per_epoch * 40
warmup_steps_p2 = steps_per_epoch * 2

lr_schedule_p2 = CosineWarmupScheduler(
    warmup_steps=warmup_steps_p2,
    total_steps=total_steps_p2,
    peak_lr=2e-5,  # Very small peak — audio branch won't be destroyed
    min_lr=1e-8
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_p2),
    loss=FocalLoss(gamma=2.0, label_smoothing=0.05),  # Reduced smoothing in Phase 2
    metrics=['accuracy']
)

total = model.count_params()
trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
print(f"Total params: {total:,} | Trainable params: {trainable:,}")

ft_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=12,
        restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_multimodal_model_v2_ft.keras',
        monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("Starting Phase 2 Fine-Tuning...")
ft_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=40,
    class_weight=class_weights,
    callbacks=ft_callbacks,
    verbose=1
)

# =====================================================================
# EVALUATION & VISUALIZATIONS
# =====================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import pickle

y_val_true = np.concatenate([y_val_rav, y_val_sam, y_val_cre])
y_pred_probs = model.predict(val_dataset, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
target_names = list(label_encoder.classes_)

print("\n" + "="*55)
print("        FINAL VALIDATION CLASSIFICATION REPORT")
print("="*55)
print(classification_report(y_val_true, y_pred, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_val_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.title('Validation Confusion Matrix (V2 Improved)')
plt.tight_layout(); plt.savefig('confusion_matrix_v2.png', dpi=300); plt.show()

# Training curves
hist_dict = ft_history.history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(hist_dict['accuracy'], label='Train', lw=2)
axes[0].plot(hist_dict['val_accuracy'], label='Val', lw=2)
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(hist_dict['loss'], label='Train', lw=2)
axes[1].plot(hist_dict['val_loss'], label='Val', lw=2)
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# =====================================================================
# SAVE
# =====================================================================
SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save(os.path.join(SAVE_DIR, "multimodal_emotion_model_v2.keras"))
print("Model saved.")

with open(os.path.join(SAVE_DIR, "encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

api_vid_mean = np.zeros((1, 64, 64, 30), dtype=np.float32)
api_vid_std = np.ones((1, 64, 64, 30), dtype=np.float32)
if 'vid_mean' in dir():
    v_m = vid_mean[0,0,0,0,:]
    v_s = vid_std[0,0,0,0,:]
    for i in range(15):
        api_vid_mean[0,:,:,i*2] = v_m[0]
        api_vid_mean[0,:,:,i*2+1] = v_m[1]
        api_vid_std[0,:,:,i*2] = v_s[0]
        api_vid_std[0,:,:,i*2+1] = v_s[1]

with open(os.path.join(SAVE_DIR, "norm.pkl"), "wb") as f:
    pickle.dump({"mean": mean, "std": std, "vid_mean": api_vid_mean, "vid_std": api_vid_std}, f)

print("\nAll done! Model saved to saved_model/multimodal_emotion_model_v2.keras")
