import os

# Environment / threading config for CPU training
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["TF_NUM_INTRAOP_THREADS"] = "12"
os.environ["TF_NUM_INTEROP_THREADS"] = "12"

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

tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(12)

print("CPU parallelism configured (12 threads). Running in float32 (no mixed precision on CPU).")

def download_ravdness_dataset(dest_dir="./RAVDNESS_Dataset"):
    import os
    import requests

    os.makedirs(dest_dir, exist_ok=True)
    record_id = "1188976"
    base_url = f"https://zenodo.org/records/{record_id}/files"

    print("Checking and downloading RAVDESS speech video actor packages...")
    for actor in range(1, 25):
        actor_str = str(actor).zfill(2)
        filename = f"Video_Speech_Actor_{actor_str}.zip"
        file_url = f"{base_url}/{filename}?download=1"
        file_path = os.path.join(dest_dir, filename)

        if os.path.exists(file_path) and os.path.getsize(file_path) > 10 * 1024 * 1024:
            continue

        print(f"Downloading {filename}...")
        try:
            response = requests.get(file_url, stream=True, timeout=120)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded {filename} successfully.")
        except Exception as e:
            print(f"Failed to download {filename}: {e}. Rerun this cell later to resume.")

download_ravdness_dataset("./RAVDNESS_Dataset")
import os
import zipfile

dataset_dir = "./RAVDNESS_Dataset"

print(f"Checking ZIP files in: {dataset_dir}")
zip_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.zip')])

if len(zip_files) == 0:
    print("No ZIP files found in this folder. Verify the path.")
else:
    print(f"Found {len(zip_files)} ZIP files. Starting extraction...")
    for file in zip_files:
        zip_path = os.path.join(dataset_dir, file)
        actor_name = file.replace("Video_Speech_", "").replace(".zip", "")
        actor_path = os.path.join(dataset_dir, actor_name)

        if os.path.exists(actor_path):
            print(f"{actor_name} folder already exists. Skipping extraction.")
            continue

        print(f"Extracting {file}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            print(f"Finished extracting {file}")
        except Exception as e:
            print(f"Error extracting {file}: {e}")

    print("Extraction complete!")
import os

dataset_dir = "./RAVDNESS_Dataset"

print("=========================================================")
print("RAVDESS Dataset Extraction Integrity Report")
print("=========================================================")

missing_folders = []
incomplete_folders = []
total_videos = 0

for actor in range(1, 25):
    actor_str = str(actor).zfill(2)
    actor_folder_name = f"Actor_{actor_str}"
    actor_path = os.path.join(dataset_dir, actor_folder_name)

    if not os.path.exists(actor_path):
        missing_folders.append(actor_folder_name)
        print(f"{actor_folder_name}: MISSING (Not extracted)")
    else:
        videos = [f for f in os.listdir(actor_path) if f.endswith('.mp4')]
        video_count = len(videos)
        total_videos += video_count

        if video_count < 60:
            incomplete_folders.append((actor_folder_name, video_count))
            print(f"{actor_folder_name}: INCOMPLETE ({video_count}/60 files found)")
        else:
            print(f"{actor_folder_name}: OK ({video_count} videos)")

print("=========================================================")
print(f"Total video files ready for training: {total_videos}")

if len(missing_folders) == 0 and len(incomplete_folders) == 0:
    print("SUCCESS: All 24 Actors are 100% extracted and complete!")
else:
    if missing_folders:
        print(f"Missing folders (need extraction): {missing_folders}")
    if incomplete_folders:
        print("Incomplete folders (need redownload/re-extract):")
        for folder, count in incomplete_folders:
            print(f"   - {folder}: only has {count} files (should be 60)")
print("=========================================================")
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
BATCH_SIZE = 128

TARGET_AUDIO_SHAPE = (MAX_FRAMES, N_MELS + N_MFCC, 1)
TARGET_VIDEO_SHAPE = (15, 64, 64, 2)

emotion_map = {
    "ANG": "Angry", "DIS": "Disgust", "FEA": "Fear",
    "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad"
}
ravdess_emotions = {
    "01": "Neutral", "02": "Neutral", "03": "Happy", "04": "Sad",
    "05": "Angry",   "06": "Fear",    "07": "Disgust"
}
samm_emotions_map = {
    "Happiness": "Happy", "Sadness": "Sad", "Anger": "Angry",
    "Fear": "Fear", "Disgust": "Disgust"
}
EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
NUM_CLASSES = 6
print(f"6-class config loaded: {EMOTION_CLASSES}")

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

        mel = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel)
        mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features = np.concatenate((mel_db, mfcc), axis=0)
        features = features.T
        if features.shape[0] > MAX_FRAMES:
            features = features[:MAX_FRAMES, :]
        else:
            pad = MAX_FRAMES - features.shape[0]
            features = np.pad(features, ((0, pad), (0, 0)))
        features = np.nan_to_num(features)
        features = np.clip(features, -100.0, 100.0)
        return np.expand_dims(features, axis=-1)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(TARGET_AUDIO_SHAPE, dtype=np.float32)


def process_audio_file(file_path):
    signal = load_audio(file_path)
    return extract_audio_features(signal)
import subprocess

def extract_landmark_masked_flow_from_frames(frame_list, target_frames=16, img_size=(64, 64)):
    if len(frame_list) < 2:
        return np.zeros(TARGET_VIDEO_SHAPE, dtype=np.float32)

    frames = []
    for frame in frame_list:
        if frame is None:
            continue
        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            h, w = gray.shape
            y_start_eye, y_end_eye = int(h * 0.15), int(h * 0.45)
            x_start_eye, x_end_eye = int(w * 0.2), int(w * 0.8)
            y_start_mouth, y_end_mouth = int(h * 0.65), int(h * 0.9)
            x_start_mouth, x_end_mouth = int(w * 0.25), int(w * 0.75)

            eyes_brow = gray[y_start_eye:y_end_eye, x_start_eye:x_end_eye]
            mouth = gray[y_start_mouth:y_end_mouth, x_start_mouth:x_end_mouth]

            if eyes_brow.size == 0 or mouth.size == 0:
                continue

            eyes_resized = cv2.resize(eyes_brow, (img_size[0], img_size[1] // 2))
            mouth_resized = cv2.resize(mouth, (img_size[0], img_size[1] // 2))
            combined_roi = np.vstack([eyes_resized, mouth_resized])
            frames.append(combined_roi)
        except Exception:
            continue
    if len(frames) < 2:
        return np.zeros(TARGET_VIDEO_SHAPE, dtype=np.float32)

    indices = np.linspace(0, len(frames) - 1, target_frames).astype(int)
    selected_frames = [frames[i] for i in indices]
    flow_sequence = []

    for i in range(len(selected_frames) - 1):
        prev = selected_frames[i]
        nxt = selected_frames[i + 1]
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev, nxt, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            flow = np.clip(flow, -50.0, 50.0)
            flow = np.nan_to_num(flow)
            flow_sequence.append(flow)
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
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
    finally:
        cap.release()
    return extract_landmark_masked_flow_from_frames(frames)


def extract_audio_track(video_path, temp_audio):
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "22050", "-ac", "1", temp_audio],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60
        )
        return result.returncode == 0 and os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0
    except Exception:
        return False

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

N_WORKERS = max(1, multiprocessing.cpu_count() - 1)

def _process_ravdess_file(args):
    idx, video_path, emotion_code, actor_id = args
    temp_audio = f"temp_ravdess_{idx}_{os.getpid()}.wav"
    try:
        ok = extract_audio_track(video_path, temp_audio)
        audio_feat = process_audio_file(temp_audio) if ok else np.zeros(TARGET_AUDIO_SHAPE, dtype=np.float32)
        video_feat = extract_flow_from_video(video_path)

        if audio_feat is None or video_feat is None:
            return None
        if np.isnan(audio_feat).any() or np.isnan(video_feat).any():
            return None
        return (audio_feat, video_feat, ravdess_emotions[emotion_code], actor_id)
    except Exception as e:
        print(f"Failed to process sample {video_path}: {e}")
        return None
    finally:
        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except OSError:
                pass


def build_ravdness_dataset(dataset_dir, cache_file="ravdess_features.npz"):
    if os.path.exists(cache_file):
        print(f"Loading pre-extracted RAVDESS features from {cache_file}...", flush=True)
        data = np.load(cache_file)
        return data['aud'], data['vid'], data['y'], data['actors']

    print(f"Scanning RAVDESS directory: {dataset_dir}", flush=True)
    if not os.path.exists(dataset_dir):
        print(f"Error: RAVDESS directory not found at {dataset_dir}", flush=True)
        return (np.empty((0, *TARGET_AUDIO_SHAPE)), np.empty((0, *TARGET_VIDEO_SHAPE)),
                np.array([]), np.array([]))

    tasks = []
    idx = 0
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.mp4'):
                parts = file.split('-')
                if len(parts) < 7:
                    continue
                emotion_code = parts[2]
                actor_id = parts[6].split('.')[0]
                if emotion_code in ravdess_emotions:
                    video_path = os.path.join(root, file)
                    if os.path.getsize(video_path) == 0:
                        continue
                    tasks.append((idx, video_path, emotion_code, actor_id))
                    idx += 1

    print(f"Dispatching {len(tasks)} RAVDESS files across {N_WORKERS} worker processes...", flush=True)
    X_audio, X_video, y, actors = [], [], [], []
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(_process_ravdess_file, t) for t in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result is not None:
                aud, vid, label, actor_id = result
                X_audio.append(aud)
                X_video.append(vid)
                y.append(label)
                actors.append(actor_id)
            if done % 200 == 0:
                print(f"  Processed {done}/{len(tasks)} RAVDESS video files...", flush=True)

    aud_arr, vid_arr = np.array(X_audio, dtype=np.float32), np.array(X_video, dtype=np.float32)
    y_arr, act_arr = np.array(y), np.array(actors)

    if len(aud_arr) > 0:
        np.savez_compressed(cache_file, aud=aud_arr, vid=vid_arr, y=y_arr, actors=act_arr)
        print(f"Saved RAVDESS features to {cache_file} for instant loading next time.", flush=True)

    return aud_arr, vid_arr, y_arr, act_arr


def _process_samm_row(args):
    subject_id, filename, emotion, dataset_dir = args
    try:
        video_file = f"{subject_id}_{filename}.mp4"
        video_path = os.path.join(dataset_dir, subject_id, video_file)
        folder_path = os.path.join(dataset_dir, subject_id, filename)

        flow = None
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            flow = extract_flow_from_video(video_path)
        elif os.path.isdir(folder_path):
            image_files = sorted([
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            frames = [cv2.imread(img_p) for img_p in image_files]
            frames = [f for f in frames if f is not None]
            if len(frames) > 0:
                flow = extract_landmark_masked_flow_from_frames(frames)

        if flow is not None and not np.isnan(flow).any():
            return (flow, samm_emotions_map[emotion], subject_id)
        return None
    except Exception as e:
        print(f"Failed to process SAMM row {subject_id}/{filename}: {e}")
        return None


def build_samm_dataset(dataset_dir, excel_path, cache_file="samm_features.npz"):
    if os.path.exists(cache_file):
        print(f"Loading pre-extracted SAMM features from {cache_file}...", flush=True)
        data = np.load(cache_file)
        return data['vid'], data['y'], data['subjects']

    if not os.path.exists(excel_path):
        print(f"Error: SAMM Excel file not found at {excel_path}", flush=True)
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
            if 'subject' in col_str:
                col_mapping[col] = 'Subject'
            elif 'filename' in col_str:
                col_mapping[col] = 'Filename'
            elif 'emotion' in col_str:
                col_mapping[col] = 'Emotion'
        df = df.rename(columns=col_mapping)
    except Exception as e:
        print(f"Error opening Excel file: {e}", flush=True)
        return np.empty((0, *TARGET_VIDEO_SHAPE)), np.array([]), np.array([])

    df = df.dropna(subset=['Subject', 'Filename', 'Emotion'])

    tasks = []
    for _, row in df.iterrows():
        emotion = str(row['Emotion']).strip()
        if emotion not in samm_emotions_map:
            continue
        subject_id = str(row['Subject']).zfill(3)
        filename = str(row['Filename']).strip()
        tasks.append((subject_id, filename, emotion, dataset_dir))

    print(f"Dispatching {len(tasks)} SAMM rows across {N_WORKERS} worker processes...", flush=True)
    X_video, y, subjects = [], [], []
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(_process_samm_row, t) for t in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result is not None:
                flow, label, subject_id = result
                X_video.append(flow)
                y.append(label)
                subjects.append(subject_id)
            if done % 100 == 0:
                print(f"  Processed {done}/{len(tasks)} SAMM rows...", flush=True)

    vid_arr, y_arr, sub_arr = np.array(X_video, dtype=np.float32), np.array(y), np.array(subjects)
    if len(vid_arr) > 0:
        np.savez_compressed(cache_file, vid=vid_arr, y=y_arr, subjects=sub_arr)
        print(f"Saved SAMM features to {cache_file} for instant loading next time.", flush=True)

    return vid_arr, y_arr, sub_arr


def _process_cremad_file(args):
    fname, dataset_path = args
    cremad_emotion_map = {
        "ANG": "Angry", "DIS": "Disgust", "FEA": "Fear",
        "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad"
    }
    parts = fname.replace('.wav', '').split('_')
    if len(parts) < 3:
        return None
    actor_id = parts[0]
    emotion_code = parts[2]
    if emotion_code not in cremad_emotion_map:
        return None

    file_path = os.path.join(dataset_path, fname)
    try:
        signal = load_audio(file_path)
        features = extract_audio_features(signal)
        if features is None or features.shape != TARGET_AUDIO_SHAPE:
            return None
        if not np.isfinite(features).all():
            return None
        return (features, cremad_emotion_map[emotion_code], f"cre_{actor_id}")
    except Exception:
        return None


def build_cremad_dataset(dataset_path="./AudioWAV", cache_file="cremad_features.npz"):
    if os.path.exists(cache_file):
        print(f"Loading pre-extracted CREMA-D features from {cache_file}...", flush=True)
        data = np.load(cache_file, allow_pickle=True)
        return data['aud'], data['y'], data['actors']

    if not os.path.exists(dataset_path):
        print(f"CREMA-D not found at {dataset_path}. Skipping.")
        return None, None, None

    wav_files = [f for f in os.listdir(dataset_path) if f.lower().endswith('.wav')]
    print(f"CREMA-D: Found {len(wav_files)} WAV files in {dataset_path}")
    print(f"Dispatching across {N_WORKERS} worker processes...", flush=True)

    tasks = [(f, dataset_path) for f in wav_files]
    X_audio, y_labels, actor_ids = [], [], []
    skipped = 0
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(_process_cremad_file, t) for t in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result is not None:
                feat, label, actor_id = result
                X_audio.append(feat)
                y_labels.append(label)
                actor_ids.append(actor_id)
            else:
                skipped += 1
            if done % 1500 == 0:
                print(f"  Processed {done}/{len(tasks)}...")

    X_audio = np.array(X_audio, dtype=np.float32)
    y_labels = np.array(y_labels)
    actor_ids = np.array(actor_ids)

    print(f"CREMA-D loaded: {len(X_audio)} samples (skipped {skipped})")

    if len(X_audio) > 0:
        np.savez_compressed(cache_file, aud=X_audio, y=y_labels, actors=actor_ids)
        print(f"Saved CREMA-D features to {cache_file} for instant loading next time.", flush=True)

    return X_audio, y_labels, actor_ids

if __name__ == '__main__':
    print("Loading datasets...", flush=True)
    X_audio_rav, X_video_rav, y_rav, actors_rav = build_ravdness_dataset("./RAVDNESS_Dataset")
    X_video_sam, y_sam, subjects_sam = build_samm_dataset(
        "./SAMM/SAMM-full/SAMM-full/SAMM",
        "./SAMM/SAMM_Micro_FACS_Codes_v2.xlsx"
    )
    X_audio_cre, y_cre, actors_cre = build_cremad_dataset("./AudioWAV")

    label_encoder = LabelEncoder()
    label_encoder.fit(["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"])
    num_classes = 6
    print(f"Classes: {list(label_encoder.classes_)}", flush=True)

    # RAVDESS split by actor
    actors_rav = np.array([f"rav_{a}" for a in actors_rav])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gss.split(X_audio_rav, y_rav, groups=actors_rav))
    X_train_rav_aud = X_audio_rav[train_idx]
    X_train_rav_vid = X_video_rav[train_idx]
    X_val_rav_aud   = X_audio_rav[val_idx]
    X_val_rav_vid   = X_video_rav[val_idx]
    y_train_rav = label_encoder.transform(y_rav[train_idx])
    y_val_rav   = label_encoder.transform(y_rav[val_idx])

    # SAMM split by subject
    subjects_sam = np.array([f"sam_{s}" for s in subjects_sam])
    gss_sam = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    s_train_idx, s_val_idx = next(gss_sam.split(X_video_sam, y_sam, groups=subjects_sam))
    X_train_sam_vid = X_video_sam[s_train_idx]
    X_val_sam_vid   = X_video_sam[s_val_idx]
    y_train_sam = label_encoder.transform(y_sam[s_train_idx])
    y_val_sam   = label_encoder.transform(y_sam[s_val_idx])

    # CREMA-D split by actor
    gss_cre = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    c_train_idx, c_val_idx = next(gss_cre.split(X_audio_cre, y_cre, groups=actors_cre))
    X_train_cre_aud = X_audio_cre[c_train_idx]
    X_val_cre_aud   = X_audio_cre[c_val_idx]
    y_train_cre = label_encoder.transform(y_cre[c_train_idx])
    y_val_cre   = label_encoder.transform(y_cre[c_val_idx])

    # Class weights
    y_train_all = np.concatenate([y_train_rav, y_train_sam, y_train_cre])
    weights = compute_class_weight("balanced", classes=np.unique(y_train_all), y=y_train_all)
    class_weights = dict(zip(np.unique(y_train_all), weights))
    disgust_idx = int(np.where(label_encoder.classes_ == "Disgust")[0][0])
    class_weights[disgust_idx] *= 0.6
    print(f"Ready! RAV:{len(y_train_rav)} SAM:{len(y_train_sam)} CRE:{len(y_train_cre)}", flush=True)

    import gc
    # 1. Audio Normalization
    temp_aud = np.concatenate([X_train_rav_aud, X_train_cre_aud], axis=0)
    mean = temp_aud.mean(axis=(0, 1, 3), keepdims=True)
    std  = temp_aud.std(axis=(0, 1, 3), keepdims=True) + 1e-6
    del temp_aud
    gc.collect()

    X_train_rav_aud = (X_train_rav_aud - mean) / std
    X_train_cre_aud = (X_train_cre_aud - mean) / std
    X_val_rav_aud   = (X_val_rav_aud   - mean) / std
    X_val_cre_aud   = (X_val_cre_aud   - mean) / std
    print("Audio normalized in-place.")

    # 2. Video Normalization
    temp_vid = np.concatenate([X_train_rav_vid, X_train_sam_vid], axis=0)
    vid_mean = temp_vid.mean(axis=(0, 1, 2, 3), keepdims=True)
    vid_std  = temp_vid.std(axis=(0, 1, 2, 3), keepdims=True) + 1e-6
    del temp_vid
    gc.collect()

    X_train_rav_vid = (X_train_rav_vid - vid_mean) / vid_std
    X_train_sam_vid = (X_train_sam_vid - vid_mean) / vid_std
    X_val_rav_vid   = (X_val_rav_vid   - vid_mean) / vid_std
    X_val_sam_vid   = (X_val_sam_vid   - vid_mean) / vid_std
    print("Video normalized in-place.")

    def prereshape_video(arr):
        n = arr.shape[0]
        return np.transpose(arr, (0, 2, 3, 1, 4)).reshape(n, 64, 64, 30).astype(np.float32)

    X_train_rav_vid = prereshape_video(X_train_rav_vid)
    X_val_rav_vid   = prereshape_video(X_val_rav_vid)
    X_train_sam_vid = prereshape_video(X_train_sam_vid)
    X_val_sam_vid   = prereshape_video(X_val_sam_vid)
    print("Video pre-reshaped to (N, 64, 64, 30).")
    TARGET_VIDEO_SHAPE = (64, 64, 30)

    # =====================================================================
    # 🚀 SUPER-FAST NUMPY TRAINING (Bypasses tf.data overhead)
    # =====================================================================
    print("Concatenating all datasets into RAM for instant CPU training...")
    n_sam = len(y_train_sam)
    X_train_sam_aud = np.zeros((n_sam, 150, 136, 1), dtype=np.float32)
    n_cre = len(y_train_cre)
    X_train_cre_vid = np.zeros((n_cre, 64, 64, 30), dtype=np.float32)

    X_train_audio_full = np.concatenate([X_train_rav_aud, X_train_sam_aud, X_train_cre_aud], axis=0)
    X_train_video_full = np.concatenate([X_train_rav_vid, X_train_sam_vid, X_train_cre_vid], axis=0)
    y_train_full       = np.concatenate([y_train_rav,     y_train_sam,     y_train_cre],     axis=0)

    n_sam_val = len(y_val_sam)
    X_val_sam_aud = np.zeros((n_sam_val, 150, 136, 1), dtype=np.float32)
    n_cre_val = len(y_val_cre)
    X_val_cre_vid = np.zeros((n_cre_val, 64, 64, 30), dtype=np.float32)

    X_val_audio_full = np.concatenate([X_val_rav_aud, X_val_sam_aud, X_val_cre_aud], axis=0)
    X_val_video_full = np.concatenate([X_val_rav_vid, X_val_sam_vid, X_val_cre_vid], axis=0)
    y_val_full       = np.concatenate([y_val_rav,     y_val_sam,     y_val_cre],     axis=0)

    y_train_cat = to_categorical(y_train_full, num_classes=num_classes)
    y_val_cat   = to_categorical(y_val_full, num_classes=num_classes)
    print(f"Total Training Samples: {len(y_train_full)}")
    print("⚡ Data is ready in pure NumPy! Expected: ~10s per epoch")

    L2 = regularizers.l2(5e-4)

    # --- BRANCH 1: AUDIO ---
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
    features = x_aud.shape[2] * x_aud.shape[3]
    x_aud = layers.Reshape((time_steps, features), name="audio_reshape")(x_aud)
    x_aud = layers.Conv1D(64, 1, padding='same', activation='relu', kernel_regularizer=L2, name="audio_conv1d_1")(x_aud)
    x_aud = layers.BatchNormalization(name="audio_bn_5")(x_aud)
    x_aud = layers.Dropout(0.3, name="audio_drop_4")(x_aud)
    x_aud = layers.Conv1D(32, 3, padding='same', activation='relu', kernel_regularizer=L2, name="audio_conv1d_2")(x_aud)
    x_aud = layers.BatchNormalization(name="audio_bn_6")(x_aud)
    x_aud = layers.GlobalAveragePooling1D(name="audio_gap1d")(x_aud)
    audio_emb = layers.Dense(128, activation='relu', kernel_regularizer=L2, name="audio_dense")(x_aud)

    # --- BRANCH 2: VIDEO ---
    video_inputs = layers.Input(shape=TARGET_VIDEO_SHAPE, name="video_input")
    x_vid = layers.SeparableConv2D(32, (3,3), padding='same', kernel_regularizer=L2, name="video_conv_1")(video_inputs)
    x_vid = layers.BatchNormalization(name="video_bn_1")(x_vid)
    x_vid = layers.Activation('relu', name="video_act_1")(x_vid)
    x_vid = layers.MaxPooling2D((2,2), name="video_pool_1")(x_vid)
    x_vid = layers.Dropout(0.25, name="video_drop_1")(x_vid)

    x_vid = layers.SeparableConv2D(64, (3,3), padding='same', kernel_regularizer=L2, name="video_conv_2")(x_vid)
    x_vid = layers.BatchNormalization(name="video_bn_2")(x_vid)
    x_vid = layers.Activation('relu', name="video_act_2")(x_vid)
    x_vid = layers.MaxPooling2D((2,2), name="video_pool_2")(x_vid)
    x_vid = layers.Dropout(0.35, name="video_drop_2")(x_vid)

    x_vid = layers.SeparableConv2D(128, (3,3), padding='same', kernel_regularizer=L2, name="video_conv_3")(x_vid)
    x_vid = layers.BatchNormalization(name="video_bn_3")(x_vid)
    x_vid = layers.Activation('relu', name="video_act_3")(x_vid)
    x_vid = layers.Dropout(0.4, name="video_drop_3")(x_vid)

    x_vid = layers.GlobalAveragePooling2D(name="video_gap")(x_vid)
    video_emb = layers.Dense(128, activation='relu', kernel_regularizer=L2, name="video_dense")(x_vid)

    # --- SQUEEZE-AND-EXCITATION (SE) FUSION ---
    audio_seq = layers.Reshape((1, 128), name="audio_seq")(audio_emb)
    video_seq = layers.Reshape((1, 128), name="video_seq")(video_emb)
    merged_seq = layers.Concatenate(axis=1, name="fusion_seq")([audio_seq, video_seq])

    sq = layers.GlobalAveragePooling1D(name="se_squeeze")(merged_seq)
    ex = layers.Dense(32, activation='relu', name="se_ex1")(sq)
    ex = layers.Dense(128, activation='sigmoid', name="se_ex2")(ex)
    ex = layers.Reshape((1, 128), name="se_reshape")(ex)
    se_out = layers.Multiply(name="se_scale")([merged_seq, ex])

    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=128, name="cross_attention")(se_out, se_out)
    attn_out = layers.Add(name="residual_attn")([se_out, attn_out])
    attn_out = layers.Flatten(name="attn_flatten")(attn_out)

    fc = layers.Dense(128, activation='relu', kernel_regularizer=L2, name="fc_fusion_1")(attn_out)
    fc = layers.Dropout(0.4, name="fc_drop_1")(fc)
    fc = layers.Dense(64, activation='relu', kernel_regularizer=L2, name="fc_fusion_2")(fc)
    fc = layers.Dropout(0.2, name="fc_drop_2")(fc)

    outputs = layers.Dense(num_classes, activation='softmax', name="softmax_output")(fc)

    model = models.Model(inputs=[audio_inputs, video_inputs], outputs=outputs, name="Moodwave_V2")
    model.summary()

    v1_model_path = "best_model.keras"
    if not os.path.exists(v1_model_path):
        v1_model_path = "d:/Emotion Detection system/best_model.keras"

    if os.path.exists(v1_model_path):
        v1_model = tf.keras.models.load_model(v1_model_path)
        print(f"V1 Model loaded from {v1_model_path}")

        v1_to_v2_mapping = {
            "conv2d": "audio_conv2d_1",
            "batch_normalization": "audio_bn_1",
            "conv2d_1": "audio_conv2d_2",
            "batch_normalization_1": "audio_bn_2",
            "conv2d_2": "audio_conv2d_3",
            "batch_normalization_2": "audio_bn_3",
            "conv2d_3": "audio_conv2d_4",
            "batch_normalization_3": "audio_bn_4",
            "conv1d": "audio_conv1d_1",
            "batch_normalization_4": "audio_bn_5",
            "conv1d_1": "audio_conv1d_2",
            "batch_normalization_5": "audio_bn_6",
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

        print(f"Transferred {transferred}/13 audio layers.")
    else:
        print("V1 weights not found - audio branch trains from scratch.")

    class FocalLoss(tf.keras.losses.Loss):
        def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.05, **kwargs):
            super().__init__(**kwargs)
            self.gamma = gamma
            self.alpha = alpha
            self.label_smoothing = label_smoothing

        def call(self, y_true, y_pred):
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            ce = -y_true * tf.math.log(y_pred)
            p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
            focal_weight = tf.pow(1.0 - p_t, self.gamma)
            return tf.reduce_mean(tf.reduce_sum(focal_weight * ce, axis=-1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=FocalLoss(gamma=2.0, label_smoothing=0.05),
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.3,
            patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_multimodal_model.keras',
            monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    print("Starting Phase 1 Training with Pure NumPy Arrays...")
    history = model.fit(
        x={"audio_input": X_train_audio_full, "video_input": X_train_video_full},
        y=y_train_cat,
        validation_data=({"audio_input": X_val_audio_full, "video_input": X_val_video_full}, y_val_cat),
        epochs=28,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val', linewidth=2)
    ax1.set_title('Accuracy'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val', linewidth=2)
    ax2.set_title('Loss'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(); plt.savefig("training_curves_phase1.png")

    import pickle
    SAVE_DIR = "saved_model"
    os.makedirs(SAVE_DIR, exist_ok=True)

    model.save(os.path.join(SAVE_DIR, "multimodal_emotion_model.keras"))
    with open(os.path.join(SAVE_DIR, "encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    api_vid_mean = np.zeros((1, 64, 64, 30), dtype=np.float32)
    api_vid_std = np.ones((1, 64, 64, 30), dtype=np.float32)
    if 'vid_mean' in globals() or 'vid_mean' in locals():
        v_m = vid_mean[0, 0, 0, 0, :]
        v_s = vid_std[0, 0, 0, 0, :]
        for i in range(15):
            api_vid_mean[0, :, :, i * 2] = v_m[0]
            api_vid_mean[0, :, :, i * 2 + 1] = v_m[1]
            api_vid_std[0, :, :, i * 2] = v_s[0]
            api_vid_std[0, :, :, i * 2 + 1] = v_s[1]

    with open(os.path.join(SAVE_DIR, "norm.pkl"), "wb") as f:
        pickle.dump({
            "mean": mean, "std": std,
            "vid_mean": api_vid_mean, "vid_std": api_vid_std
        }, f)
    
    # Phase 2
    model.load_weights('best_multimodal_model.keras')
    model.trainable = True

    print("Phase 2 Unfrozen!")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=FocalLoss(gamma=2.0, label_smoothing=0.05),
        metrics=['accuracy']
    )

    ft_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.3,
            patience=3, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_multimodal_model_ft.keras',
            monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print("Starting Phase 2 Fine-Tuning...")
    ft_history = model.fit(
        x={"audio_input": X_train_audio_full, "video_input": X_train_video_full},
        y=y_train_cat,
        validation_data=({"audio_input": X_val_audio_full, "video_input": X_val_video_full}, y_val_cat),
        epochs=40,
        batch_size=64,
        class_weight=class_weights,
        callbacks=ft_callbacks,
        shuffle=True,
        verbose=1
    )
    
    model.save(os.path.join(SAVE_DIR, "multimodal_emotion_model.keras"))
    print("Serialization complete! Model ready for deployment.")
