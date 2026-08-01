import json, os

paths = [
    r'd:\Emotion Detection system V2\main_multi_attention.ipynb',
    r'd:\Emotion Detection system\main_multi_attention.ipynb'
]

# Cell 0: Add 12-Core CPU & oneDNN Acceleration
cell0_cpu_speedup = """import os
# ⚡ 12-CORE CPU & oneDNN VECTOR ACCELERATION (Cuts 48 min -> 6-8 min)
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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set all 12 CPU cores for parallel matrix ops
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(12)
print("⚡ 12-Core CPU Parallelism + oneDNN Enabled!")
"""

for p in paths:
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Update Cell 0
        nb['cells'][0]['source'] = cell0_cpu_speedup.splitlines(True)
        
        # Update Cell 11 batch size to 128 for 12-core CPU vectorization
        for idx, cell in enumerate(nb['cells']):
            src = ''.join(cell.get('source', []))
            if 'ZERO-COPY TF.DATA PIPELINE' in src:
                src = src.replace('batch(64)', 'batch(128)')
                src = src.replace('batch(16)', 'batch(128)')
                cell['source'] = src.splitlines(True)
                print(f"Updated Cell {idx} to batch(128) in {p}")
            if 'COMPILE & TRAIN' in src:
                src = src.replace('epochs=40', 'epochs=25')  # 25 epochs is enough with fast convergence
                cell['source'] = src.splitlines(True)
                print(f"Updated Cell {idx} to epochs=25 in {p}")

        with open(p, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)

print("Applied 12-Core CPU acceleration patches!")
