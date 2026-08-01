"""
MoodWave V2 — Optimized Fast Training Script
=============================================
Same model architecture as main_multi_attention.ipynb.
Only training hyperparameters and loop mechanics are changed for speed.

Targets:
  - ~1 epoch per minute on CPU (12 threads)
  - 70%+ training AND validation accuracy
  - No overfitting, no underfitting, no bias, no errors
"""

import os
import time

# =====================================================================
# ENVIRONMENT: Enable OneDNN for ~1.5-2x CPU speedup
# =====================================================================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["TF_NUM_INTRAOP_THREADS"] = "12"
os.environ["TF_NUM_INTEROP_THREADS"] = "12"

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
import warnings
import gc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(12)

# Ensure graph mode (not eager) for maximum speed
tf.config.run_functions_eagerly(False)

print("=" * 60)
print("MoodWave V2 -- Optimized Fast Training")
print("CPU parallelism: 12 threads | OneDNN: ENABLED | float32")
print("=" * 60)

# =====================================================================
# CONSTANTS (identical to notebook)
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

TARGET_AUDIO_SHAPE = (MAX_FRAMES, N_MELS + N_MFCC, 1)  # (150, 136, 1)
TARGET_VIDEO_SHAPE_RAW = (15, 64, 64, 2)
TARGET_VIDEO_SHAPE = (64, 64, 30)  # after pre-reshape

EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
NUM_CLASSES = 6

ravdess_emotions = {
    "01": "Neutral", "02": "Neutral", "03": "Happy", "04": "Sad",
    "05": "Angry",   "06": "Fear",    "07": "Disgust"
}

# =====================================================================
# OPTIMIZED HYPERPARAMETERS
# =====================================================================
BATCH_SIZE = 128          # Up from 64 — fewer steps/epoch, better CPU SIMD
MAX_EPOCHS = 35           # Single phase, cosine annealing
PATIENCE = 8              # Early stopping patience
INITIAL_LR = 5e-4         # Slightly higher starting LR
MIN_LR = 1e-6             # Cosine floor
LABEL_SMOOTHING = 0.08    # Up from 0.05 — narrows train/val gap
FOCAL_GAMMA = 2.0         # Same as notebook
DISGUST_WEIGHT_MULT = 0.5 # Down from 0.6 — less Disgust over-dominance
STEPS_PER_EXEC = 4        # Reduce Python overhead in training loop
NOISE_STD = 0.02          # Gaussian noise for audio augmentation

print(f"\nHyperparameters: BS={BATCH_SIZE}, LR={INITIAL_LR}->{MIN_LR}, "
      f"Epochs<={MAX_EPOCHS}, Patience={PATIENCE}")
print(f"Label Smoothing={LABEL_SMOOTHING}, Focal gamma={FOCAL_GAMMA}, "
      f"Noise sigma={NOISE_STD}")


# =====================================================================
# LOAD CACHED FEATURES (identical to notebook)
# =====================================================================
def load_cached_datasets():
    """Load pre-extracted features from .npz cache files."""
    print("\n--- Loading cached datasets ---")

    # RAVDESS
    rav_cache = "ravdess_features.npz"
    if not os.path.exists(rav_cache):
        raise FileNotFoundError(f"{rav_cache} not found. Run the notebook first to extract features.")
    data = np.load(rav_cache)
    X_audio_rav = data['aud']
    X_video_rav = data['vid']
    y_rav = data['y']
    actors_rav = data['actors']
    print(f"  RAVDESS: {len(y_rav)} samples loaded")

    # SAMM
    samm_cache = "samm_features.npz"
    if not os.path.exists(samm_cache):
        raise FileNotFoundError(f"{samm_cache} not found. Run the notebook first to extract features.")
    data = np.load(samm_cache)
    X_video_sam = data['vid']
    y_sam = data['y']
    subjects_sam = data['subjects']
    print(f"  SAMM:    {len(y_sam)} samples loaded")

    # CREMA-D
    cre_cache = "cremad_features.npz"
    if not os.path.exists(cre_cache):
        raise FileNotFoundError(f"{cre_cache} not found. Run the notebook first to extract features.")
    data = np.load(cre_cache, allow_pickle=True)
    X_audio_cre = data['aud']
    y_cre = data['y']
    actors_cre = data['actors']
    print(f"  CREMA-D: {len(y_cre)} samples loaded")

    return (X_audio_rav, X_video_rav, y_rav, actors_rav,
            X_video_sam, y_sam, subjects_sam,
            X_audio_cre, y_cre, actors_cre)


# =====================================================================
# SPLIT & NORMALIZE (identical logic to notebook)
# =====================================================================
def prepare_data(X_audio_rav, X_video_rav, y_rav, actors_rav,
                 X_video_sam, y_sam, subjects_sam,
                 X_audio_cre, y_cre, actors_cre):
    """Split, normalize, reshape — same logic as notebook."""

    label_encoder = LabelEncoder()
    label_encoder.fit(EMOTION_CLASSES)

    # --- RAVDESS split by actor ---
    actors_rav = np.array([f"rav_{a}" for a in actors_rav])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gss.split(X_audio_rav, y_rav, groups=actors_rav))
    X_train_rav_aud = X_audio_rav[train_idx]
    X_train_rav_vid = X_video_rav[train_idx]
    X_val_rav_aud   = X_audio_rav[val_idx]
    X_val_rav_vid   = X_video_rav[val_idx]
    y_train_rav = label_encoder.transform(y_rav[train_idx])
    y_val_rav   = label_encoder.transform(y_rav[val_idx])

    # --- SAMM split by subject ---
    subjects_sam = np.array([f"sam_{s}" for s in subjects_sam])
    gss_sam = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    s_train_idx, s_val_idx = next(gss_sam.split(X_video_sam, y_sam, groups=subjects_sam))
    X_train_sam_vid = X_video_sam[s_train_idx]
    X_val_sam_vid   = X_video_sam[s_val_idx]
    y_train_sam = label_encoder.transform(y_sam[s_train_idx])
    y_val_sam   = label_encoder.transform(y_sam[s_val_idx])

    # --- CREMA-D split by actor ---
    gss_cre = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    c_train_idx, c_val_idx = next(gss_cre.split(X_audio_cre, y_cre, groups=actors_cre))
    X_train_cre_aud = X_audio_cre[c_train_idx]
    X_val_cre_aud   = X_audio_cre[c_val_idx]
    y_train_cre = label_encoder.transform(y_cre[c_train_idx])
    y_val_cre   = label_encoder.transform(y_cre[c_val_idx])

    # --- Class weights (identical logic) ---
    y_train_all = np.concatenate([y_train_rav, y_train_sam, y_train_cre])
    weights = compute_class_weight("balanced", classes=np.unique(y_train_all), y=y_train_all)
    class_weights = dict(zip(np.unique(y_train_all), weights))
    disgust_idx = int(np.where(label_encoder.classes_ == "Disgust")[0][0])
    class_weights[disgust_idx] *= DISGUST_WEIGHT_MULT
    print(f"\nClass weights: { {label_encoder.classes_[k]: round(v,3) for k,v in class_weights.items()} }")

    # --- Audio Normalization (identical to notebook) ---
    temp_aud = np.concatenate([X_train_rav_aud, X_train_cre_aud], axis=0)
    aud_mean = temp_aud.mean(axis=(0, 1, 3), keepdims=True)
    aud_std  = temp_aud.std(axis=(0, 1, 3), keepdims=True) + 1e-6
    del temp_aud
    gc.collect()

    X_train_rav_aud = (X_train_rav_aud - aud_mean) / aud_std
    X_train_cre_aud = (X_train_cre_aud - aud_mean) / aud_std
    X_val_rav_aud   = (X_val_rav_aud   - aud_mean) / aud_std
    X_val_cre_aud   = (X_val_cre_aud   - aud_mean) / aud_std
    print("Audio normalized.")

    # --- Video Normalization (identical to notebook) ---
    temp_vid = np.concatenate([X_train_rav_vid, X_train_sam_vid], axis=0)
    vid_mean = temp_vid.mean(axis=(0, 1, 2, 3), keepdims=True)
    vid_std  = temp_vid.std(axis=(0, 1, 2, 3), keepdims=True) + 1e-6
    del temp_vid
    gc.collect()

    X_train_rav_vid = (X_train_rav_vid - vid_mean) / vid_std
    X_train_sam_vid = (X_train_sam_vid - vid_mean) / vid_std
    X_val_rav_vid   = (X_val_rav_vid   - vid_mean) / vid_std
    X_val_sam_vid   = (X_val_sam_vid   - vid_mean) / vid_std
    print("Video normalized.")

    # --- Pre-reshape video: (N, 15, 64, 64, 2) → (N, 64, 64, 30) ---
    def prereshape_video(arr):
        n = arr.shape[0]
        return np.transpose(arr, (0, 2, 3, 1, 4)).reshape(n, 64, 64, 30).astype(np.float32)

    X_train_rav_vid = prereshape_video(X_train_rav_vid)
    X_val_rav_vid   = prereshape_video(X_val_rav_vid)
    X_train_sam_vid = prereshape_video(X_train_sam_vid)
    X_val_sam_vid   = prereshape_video(X_val_sam_vid)
    print("Video reshaped to (N, 64, 64, 30).")

    # --- Concatenate all datasets ---
    n_sam = len(y_train_sam)
    X_train_sam_aud = np.zeros((n_sam, 150, 136, 1), dtype=np.float32)
    n_cre = len(y_train_cre)
    X_train_cre_vid = np.zeros((n_cre, 64, 64, 30), dtype=np.float32)

    X_train_audio = np.concatenate([X_train_rav_aud, X_train_sam_aud, X_train_cre_aud], axis=0)
    X_train_video = np.concatenate([X_train_rav_vid, X_train_sam_vid, X_train_cre_vid], axis=0)
    y_train       = np.concatenate([y_train_rav, y_train_sam, y_train_cre], axis=0)

    n_sam_val = len(y_val_sam)
    X_val_sam_aud = np.zeros((n_sam_val, 150, 136, 1), dtype=np.float32)
    n_cre_val = len(y_val_cre)
    X_val_cre_vid = np.zeros((n_cre_val, 64, 64, 30), dtype=np.float32)

    X_val_audio = np.concatenate([X_val_rav_aud, X_val_sam_aud, X_val_cre_aud], axis=0)
    X_val_video = np.concatenate([X_val_rav_vid, X_val_sam_vid, X_val_cre_vid], axis=0)
    y_val       = np.concatenate([y_val_rav, y_val_sam, y_val_cre], axis=0)

    y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_cat   = to_categorical(y_val, num_classes=NUM_CLASSES)

    print(f"\nTotal: Train={len(y_train)}, Val={len(y_val)}")

    # Free intermediate arrays
    del X_train_rav_aud, X_train_rav_vid, X_train_sam_aud, X_train_sam_vid
    del X_train_cre_aud, X_train_cre_vid
    del X_val_rav_aud, X_val_rav_vid, X_val_sam_aud, X_val_sam_vid
    del X_val_cre_aud, X_val_cre_vid
    gc.collect()

    return (X_train_audio, X_train_video, y_train_cat,
            X_val_audio, X_val_video, y_val_cat, y_val,
            class_weights, label_encoder,
            aud_mean, aud_std, vid_mean, vid_std)


# =====================================================================
# MODEL (IDENTICAL architecture to notebook — byte-for-byte)
# =====================================================================
def build_model():
    """Build the exact same Moodwave_V2 architecture as the notebook."""
    L2 = regularizers.l2(5e-4)

    # --- BRANCH 1: AUDIO (identical) ---
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

    # --- BRANCH 2: VIDEO (identical) ---
    video_inputs = layers.Input(shape=TARGET_VIDEO_SHAPE, name="video_input")
    x_vid = layers.SeparableConv2D(32, (3,3), padding='same', depthwise_regularizer=L2, pointwise_regularizer=L2, name="video_conv_1")(video_inputs)
    x_vid = layers.BatchNormalization(name="video_bn_1")(x_vid)
    x_vid = layers.Activation('relu', name="video_act_1")(x_vid)
    x_vid = layers.MaxPooling2D((2,2), name="video_pool_1")(x_vid)
    x_vid = layers.Dropout(0.25, name="video_drop_1")(x_vid)

    x_vid = layers.SeparableConv2D(64, (3,3), padding='same', depthwise_regularizer=L2, pointwise_regularizer=L2, name="video_conv_2")(x_vid)
    x_vid = layers.BatchNormalization(name="video_bn_2")(x_vid)
    x_vid = layers.Activation('relu', name="video_act_2")(x_vid)
    x_vid = layers.MaxPooling2D((2,2), name="video_pool_2")(x_vid)
    x_vid = layers.Dropout(0.35, name="video_drop_2")(x_vid)

    x_vid = layers.SeparableConv2D(128, (3,3), padding='same', depthwise_regularizer=L2, pointwise_regularizer=L2, name="video_conv_3")(x_vid)
    x_vid = layers.BatchNormalization(name="video_bn_3")(x_vid)
    x_vid = layers.Activation('relu', name="video_act_3")(x_vid)
    x_vid = layers.Dropout(0.4, name="video_drop_3")(x_vid)

    x_vid = layers.GlobalAveragePooling2D(name="video_gap")(x_vid)
    video_emb = layers.Dense(128, activation='relu', kernel_regularizer=L2, name="video_dense")(x_vid)

    # --- SQUEEZE-AND-EXCITATION (SE) FUSION (identical) ---
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

    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="softmax_output")(fc)

    model = models.Model(inputs=[audio_inputs, video_inputs], outputs=outputs, name="Moodwave_V2")
    return model


# =====================================================================
# TRANSFER V1 WEIGHTS (identical to notebook)
# =====================================================================
def transfer_v1_weights(model):
    """Transfer audio branch weights from V1 model, identical to notebook."""
    v1_model_path = "best_model.keras"
    if not os.path.exists(v1_model_path):
        v1_model_path = "d:/Emotion Detection system/best_model.keras"

    if not os.path.exists(v1_model_path):
        print("V1 weights not found — audio branch trains from scratch.")
        return

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
    del v1_model
    gc.collect()


# =====================================================================
# FOCAL LOSS (identical to notebook, with configurable label_smoothing)
# =====================================================================
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing,
        })
        return config


# =====================================================================
# COSINE ANNEALING LR SCHEDULE
# =====================================================================
class CosineAnnealingSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine annealing from initial_lr down to min_lr over total_steps."""
    def __init__(self, initial_lr, min_lr, total_steps):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * step / total))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "min_lr": self.min_lr,
            "total_steps": self.total_steps,
        }


# =====================================================================
# AUDIO AUGMENTATION (light Gaussian noise)
# =====================================================================
class AudioNoiseAugmentor:
    """Add light Gaussian noise to audio features during training."""
    def __init__(self, noise_std=0.02):
        self.noise_std = noise_std

    def augment(self, X_audio):
        noise = np.random.normal(0, self.noise_std, X_audio.shape).astype(np.float32)
        return X_audio + noise


# =====================================================================
# EPOCH TIMER CALLBACK
# =====================================================================
class EpochTimerCallback(tf.keras.callbacks.Callback):
    """Log wall time per epoch for speed verification."""
    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._epoch_start
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        print(f"  [Timer] Epoch {epoch+1} completed in {elapsed:.1f}s | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")


# =====================================================================
# SAVE MODEL + NORMALIZATION (identical to notebook)
# =====================================================================
def save_model_and_norms(model, label_encoder, aud_mean, aud_std, vid_mean, vid_std):
    """Save model, encoder, and normalization params — identical to notebook."""
    import pickle

    SAVE_DIR = "saved_model"
    os.makedirs(SAVE_DIR, exist_ok=True)

    model.save(os.path.join(SAVE_DIR, "multimodal_emotion_model.keras"))
    print("Model saved.")

    with open(os.path.join(SAVE_DIR, "encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    print("Encoder saved.")

    # api.py expects vid_mean/vid_std in the stacked shape (1, 64, 64, 30)
    api_vid_mean = np.zeros((1, 64, 64, 30), dtype=np.float32)
    api_vid_std = np.ones((1, 64, 64, 30), dtype=np.float32)
    if vid_mean is not None:
        v_m = vid_mean[0, 0, 0, 0, :]
        v_s = vid_std[0, 0, 0, 0, :]
        for i in range(15):
            api_vid_mean[0, :, :, i * 2] = v_m[0]
            api_vid_mean[0, :, :, i * 2 + 1] = v_m[1]
            api_vid_std[0, :, :, i * 2] = v_s[0]
            api_vid_std[0, :, :, i * 2 + 1] = v_s[1]

    with open(os.path.join(SAVE_DIR, "norm.pkl"), "wb") as f:
        pickle.dump({
            "mean": aud_mean, "std": aud_std,
            "vid_mean": api_vid_mean, "vid_std": api_vid_std
        }, f)
    print("Audio + Video normalization params saved.")
    print("Serialization complete! Ready for API deployment.")


# =====================================================================
# TRAINING CURVES
# =====================================================================
def plot_training_curves(history):
    """Save training curves to PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val', linewidth=2)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val', linewidth=2)
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("training_curves_optimized.png", dpi=150)
    print("Training curves saved to training_curves_optimized.png")
    plt.close()


# =====================================================================
# MAIN TRAINING PIPELINE
# =====================================================================
def main():
    total_start = time.time()

    # 1. Load cached features
    (X_audio_rav, X_video_rav, y_rav, actors_rav,
     X_video_sam, y_sam, subjects_sam,
     X_audio_cre, y_cre, actors_cre) = load_cached_datasets()

    # 2. Prepare data (split, normalize, concatenate)
    (X_train_audio, X_train_video, y_train_cat,
     X_val_audio, X_val_video, y_val_cat, y_val,
     class_weights, label_encoder,
     aud_mean, aud_std, vid_mean, vid_std) = prepare_data(
        X_audio_rav, X_video_rav, y_rav, actors_rav,
        X_video_sam, y_sam, subjects_sam,
        X_audio_cre, y_cre, actors_cre
    )

    # Free raw data
    del X_audio_rav, X_video_rav, y_rav, actors_rav
    del X_video_sam, y_sam, subjects_sam
    del X_audio_cre, y_cre, actors_cre
    gc.collect()

    # 3. Apply light audio augmentation to training data
    print("\nApplying audio augmentation (Gaussian noise sigma={})...".format(NOISE_STD))
    augmentor = AudioNoiseAugmentor(noise_std=NOISE_STD)
    X_train_audio_aug = augmentor.augment(X_train_audio)
    print("Audio augmentation applied.")

    # 4. Build model (identical architecture)
    print("\n--- Building Moodwave_V2 model ---")
    tf.keras.backend.clear_session()
    gc.collect()
    model = build_model()
    model.summary()

    # 5. Transfer V1 weights
    transfer_v1_weights(model)

    total = model.count_params()
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"Total: {total:,} | Trainable: {trainable:,} | Frozen: {total - trainable:,}")

    # 6. Compute LR schedule
    steps_per_epoch = int(np.ceil(len(y_train_cat) / BATCH_SIZE))
    total_steps = steps_per_epoch * MAX_EPOCHS
    lr_schedule = CosineAnnealingSchedule(INITIAL_LR, MIN_LR, total_steps)
    print(f"\nSteps/epoch: {steps_per_epoch}, Total steps: {total_steps}")
    print(f"LR: {INITIAL_LR} -> {MIN_LR} (cosine annealing)")

    # 7. Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy'],
        steps_per_execution=STEPS_PER_EXEC,
    )

    # 8. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_multimodal_model_optimized.keras',
            monitor='val_accuracy', save_best_only=True, verbose=1),
        EpochTimerCallback(),
    ]

    # 9. Train (single phase — direct NumPy, no generator)
    print("\n" + "=" * 60)
    print("STARTING OPTIMIZED TRAINING")
    print("=" * 60)

    history = model.fit(
        x={"audio_input": X_train_audio_aug, "video_input": X_train_video},
        y=y_train_cat,
        validation_data=(
            {"audio_input": X_val_audio, "video_input": X_val_video},
            y_val_cat
        ),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    # 10. Final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    best_train_acc = max(history.history['accuracy'])
    best_val_acc = max(history.history['val_accuracy'])
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    epochs_run = len(history.history['accuracy'])
    total_time = time.time() - total_start

    print(f"  Epochs run: {epochs_run}")
    print(f"  Best Train Acc: {best_train_acc:.4f}")
    print(f"  Best Val Acc:   {best_val_acc:.4f}")
    print(f"  Final Train Acc: {final_train_acc:.4f}")
    print(f"  Final Val Acc:   {final_val_acc:.4f}")
    print(f"  Train/Val gap:   {abs(final_train_acc - final_val_acc):.4f}")
    print(f"  Total time:      {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Avg time/epoch:  {total_time/epochs_run:.1f}s")

    # 11. Validate targets
    target_met = True
    if best_val_acc < 0.70:
        print("\n[!] WARNING: Val accuracy < 70% target!")
        target_met = False
    if best_train_acc < 0.70:
        print("[!] WARNING: Train accuracy < 70% target!")
        target_met = False
    if abs(final_train_acc - final_val_acc) > 0.10:
        print("[!] WARNING: Train/Val gap > 10% -- possible overfitting")
        target_met = False
    if target_met:
        print("\n[OK] ALL TARGETS MET!")

    # 12. Plot curves
    plot_training_curves(history)

    # 13. Save model + norms
    print("\n--- Saving model & normalization ---")
    # Load best weights before saving
    if os.path.exists('best_multimodal_model_optimized.keras'):
        model.load_weights('best_multimodal_model_optimized.keras')
        print("Best checkpoint weights loaded.")
    save_model_and_norms(model, label_encoder, aud_mean, aud_std, vid_mean, vid_std)

    print(f"\n{'=' * 60}")
    print("DONE! Model ready for API deployment.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
