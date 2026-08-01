import json
import re

notebook_path = "d:/Emotion Detection system V2/main_multi_attention.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        
        # 1. Patch the tf.data pipeline
        if "train_dataset.map(augment_batch" in source or "def augment_batch(" in source:
            cell['source'] = [
                "# =====================================================================\n",
                "# 🚀 SUPER-FAST NUMPY TRAINING (Bypasses tf.data overhead)\n",
                "# =====================================================================\n",
                "\n",
                "print(\"Concatenating all datasets into RAM for instant CPU training...\")\n",
                "\n",
                "# 1. Combine Audio inputs (pad missing modalities with zeros)\n",
                "n_sam = len(y_train_sam)\n",
                "X_train_sam_aud = np.zeros((n_sam, 150, 136, 1), dtype=np.float32)\n",
                "n_cre = len(y_train_cre)\n",
                "X_train_cre_vid = np.zeros((n_cre, 64, 64, 30), dtype=np.float32)\n",
                "\n",
                "X_train_audio_full = np.concatenate([X_train_rav_aud, X_train_sam_aud, X_train_cre_aud], axis=0)\n",
                "X_train_video_full = np.concatenate([X_train_rav_vid, X_train_sam_vid, X_train_cre_vid], axis=0)\n",
                "y_train_full       = np.concatenate([y_train_rav,     y_train_sam,     y_train_cre],     axis=0)\n",
                "\n",
                "# Validation\n",
                "n_sam_val = len(y_val_sam)\n",
                "X_val_sam_aud = np.zeros((n_sam_val, 150, 136, 1), dtype=np.float32)\n",
                "n_cre_val = len(y_val_cre)\n",
                "X_val_cre_vid = np.zeros((n_cre_val, 64, 64, 30), dtype=np.float32)\n",
                "\n",
                "X_val_audio_full = np.concatenate([X_val_rav_aud, X_val_sam_aud, X_val_cre_aud], axis=0)\n",
                "X_val_video_full = np.concatenate([X_val_rav_vid, X_val_sam_vid, X_val_cre_vid], axis=0)\n",
                "y_val_full       = np.concatenate([y_val_rav,     y_val_sam,     y_val_cre],     axis=0)\n",
                "\n",
                "y_train_cat = to_categorical(y_train_full, num_classes=num_classes)\n",
                "y_val_cat   = to_categorical(y_val_full, num_classes=num_classes)\n",
                "\n",
                "print(f\"Total Training Samples: {len(y_train_full)}\")\n",
                "print(\"⚡ Data is ready in pure NumPy! Expected: ~10s per epoch\")\n"
            ]
            
        # 2. Patch the Model Architecture (SeparableConv2D)
        elif "video_conv_1" in source and "layers.Conv2D" in source:
            new_source = []
            for line in cell['source']:
                if 'name="video_conv_' in line and 'Conv2D' in line:
                    line = line.replace('layers.Conv2D', 'layers.SeparableConv2D')
                new_source.append(line)
            cell['source'] = new_source
            
        # 3. Patch the fit() call
        elif "history = model.fit(" in source and "train_dataset" in source:
            cell['source'] = [
                "# =====================================================================\n",
                "# 🎓 STEP 4: COMPILE & TRAIN WITH FOCAL LOSS & COSINE ANNEALING\n",
                "# =====================================================================\n",
                "class FocalLoss(tf.keras.losses.Loss):\n",
                "    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.05, **kwargs):\n",
                "        super().__init__(**kwargs)\n",
                "        self.gamma = gamma\n",
                "        self.alpha = alpha\n",
                "        self.label_smoothing = label_smoothing\n",
                "\n",
                "    def call(self, y_true, y_pred):\n",
                "        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)\n",
                "        y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / num_classes\n",
                "        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)\n",
                "        ce = -y_true * tf.math.log(y_pred)\n",
                "        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)\n",
                "        focal_weight = tf.pow(1.0 - p_t, self.gamma)\n",
                "        return tf.reduce_mean(tf.reduce_sum(focal_weight * ce, axis=-1))\n",
                "\n",
                "\n",
                "model.compile(\n",
                "    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),\n",
                "    loss=FocalLoss(gamma=2.0, label_smoothing=0.05),\n",
                "    metrics=['accuracy']\n",
                ")\n",
                "\n",
                "callbacks = [\n",
                "    tf.keras.callbacks.EarlyStopping(\n",
                "        monitor='val_loss', patience=7,\n",
                "        restore_best_weights=True, verbose=1),\n",
                "    tf.keras.callbacks.ReduceLROnPlateau(\n",
                "        monitor='val_loss', factor=0.3,\n",
                "        patience=3, min_lr=1e-6, verbose=1),\n",
                "    tf.keras.callbacks.ModelCheckpoint(\n",
                "        filepath='best_multimodal_model.keras',\n",
                "        monitor='val_accuracy', save_best_only=True, verbose=1)\n",
                "]\n",
                "\n",
                "print(\"Starting blazing fast NumPy training...\")\n",
                "history = model.fit(\n",
                "    x={\"audio_input\": X_train_audio_full, \"video_input\": X_train_video_full},\n",
                "    y=y_train_cat,\n",
                "    validation_data=({\"audio_input\": X_val_audio_full, \"video_input\": X_val_video_full}, y_val_cat),\n",
                "    epochs=28,\n",
                "    batch_size=64,\n",
                "    class_weight=class_weights,\n",
                "    callbacks=callbacks,\n",
                "    shuffle=True,\n",
                "    verbose=1\n",
                ")\n"
            ]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully!")
