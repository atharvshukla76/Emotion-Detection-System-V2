import json

notebook_path = r'd:\Emotion Detection system V2\main_multi_attention.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes_log = []

for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    source_str = ''.join(cell['source'])
    original = source_str

    # ===== 1. ARCHITECTURE WIDENING =====

    # Audio Conv Layer 3: 64 -> 128
    source_str = source_str.replace(
        'Conv2D(64, (3,3), padding=\'same\', kernel_regularizer=L2, name="audio_conv2d_3")',
        'Conv2D(128, (3,3), padding=\'same\', kernel_regularizer=L2, name="audio_conv2d_3")')

    # Audio Conv Layer 4: 64 -> 128
    source_str = source_str.replace(
        'Conv2D(64, (3,3), padding=\'same\', kernel_regularizer=L2, name="audio_conv2d_4")',
        'Conv2D(128, (3,3), padding=\'same\', kernel_regularizer=L2, name="audio_conv2d_4")')

    # Video Conv Layer 3: 64 -> 128
    source_str = source_str.replace(
        'Conv2D(64, (3,3), padding=\'same\', kernel_regularizer=L2, name="video_conv_3")',
        'Conv2D(128, (3,3), padding=\'same\', kernel_regularizer=L2, name="video_conv_3")')

    # Audio Dense: 64 -> 128
    source_str = source_str.replace(
        'Dense(64, activation=\'relu\', kernel_regularizer=L2, name="audio_dense")',
        'Dense(128, activation=\'relu\', kernel_regularizer=L2, name="audio_dense")')

    # Video Dense: 64 -> 128
    source_str = source_str.replace(
        'Dense(64, activation=\'relu\', kernel_regularizer=L2, name="video_dense")',
        'Dense(128, activation=\'relu\', kernel_regularizer=L2, name="video_dense")')

    # Audio Reshape: (1, 64) -> (1, 128)
    source_str = source_str.replace(
        'Reshape((1, 64), name="audio_seq")',
        'Reshape((1, 128), name="audio_seq")')

    # Video Reshape: (1, 64) -> (1, 128)
    source_str = source_str.replace(
        'Reshape((1, 64), name="video_seq")',
        'Reshape((1, 128), name="video_seq")')

    # MultiHeadAttention key_dim: 64 -> 128
    source_str = source_str.replace(
        'MultiHeadAttention(num_heads=4, key_dim=64',
        'MultiHeadAttention(num_heads=4, key_dim=128')

    # Fusion Dropout 1: 0.5 -> 0.4
    source_str = source_str.replace(
        'Dropout(0.5, name="fc_drop_1")',
        'Dropout(0.4, name="fc_drop_1")')

    # Fusion Dropout 2: 0.3 -> 0.2
    source_str = source_str.replace(
        'Dropout(0.3, name="fc_drop_2")',
        'Dropout(0.2, name="fc_drop_2")')

    # ===== 2. SPEED OPTIMIZATIONS =====

    # Batch size: 16 -> 32
    source_str = source_str.replace('.batch(16)', '.batch(32)')

    # Shuffle buffer: 10000 -> 5000
    source_str = source_str.replace('buffer_size=10000', 'buffer_size=5000')

    # Phase 1 epochs: 80 -> 50
    source_str = source_str.replace('epochs=80,', 'epochs=50,')

    # Phase 1 EarlyStopping patience: 15 -> 10
    source_str = source_str.replace('patience=15,', 'patience=10,')

    # Phase 2 epochs: 40 -> 25
    source_str = source_str.replace('epochs=40,', 'epochs=25,')

    # Phase 2 learning rate: 2e-5 -> 5e-5
    source_str = source_str.replace('learning_rate=2e-5', 'learning_rate=5e-5')

    # ===== 3. FOCAL LOSS =====

    # Replace CategoricalCrossentropy with FocalLoss (catches both Phase 1 and Phase 2)
    source_str = source_str.replace(
        'loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2)',
        'loss=FocalLoss(gamma=2.0, label_smoothing=0.1)')

    # Inject FocalLoss class definition before FIRST model.compile (STEP 4 cell only)
    if 'STEP 4: COMPILE' in source_str and 'class FocalLoss' not in source_str:
        focal_code = (
            '\n'
            '# Focal Loss: focuses training on hard-to-classify samples\n'
            'class FocalLoss(tf.keras.losses.Loss):\n'
            '    def __init__(self, gamma=2.0, label_smoothing=0.1, **kwargs):\n'
            '        super().__init__(**kwargs)\n'
            '        self.gamma = gamma\n'
            '        self.label_smoothing = label_smoothing\n'
            '\n'
            '    def call(self, y_true, y_pred):\n'
            '        y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / tf.cast(tf.shape(y_true)[-1], tf.float32)\n'
            '        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)\n'
            '        cross_entropy = -y_true * tf.math.log(y_pred)\n'
            '        focal_weight = tf.pow(1 - y_pred, self.gamma)\n'
            '        return tf.reduce_sum(focal_weight * cross_entropy, axis=-1)\n'
            '\n'
            '    def get_config(self):\n'
            '        config = super().get_config()\n'
            '        config.update({"gamma": self.gamma, "label_smoothing": self.label_smoothing})\n'
            '        return config\n'
            '\n'
        )
        source_str = source_str.replace('model.compile(', focal_code + 'model.compile(', 1)
        changes_log.append('Added FocalLoss class definition')

    # ===== 4. MIXUP AUGMENTATION =====

    old_train_line = 'train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)'
    if old_train_line in source_str:
        mixup_code = (
            'train_dataset = train_dataset.batch(32)\n'
            '\n'
            '# Mixup: blends sample pairs to teach emotion boundaries\n'
            'def mixup_batch(inputs, labels, alpha=0.2):\n'
            '    batch_size = tf.shape(labels)[0]\n'
            '    lam = tf.random.uniform([], 0.0, alpha)\n'
            '    indices = tf.random.shuffle(tf.range(batch_size))\n'
            '    mixed_audio = lam * inputs["audio_input"] + (1 - lam) * tf.gather(inputs["audio_input"], indices)\n'
            '    mixed_video = lam * inputs["video_input"] + (1 - lam) * tf.gather(inputs["video_input"], indices)\n'
            '    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)\n'
            '    return {"audio_input": mixed_audio, "video_input": mixed_video}, mixed_labels\n'
            '\n'
            'train_dataset = train_dataset.map(mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)\n'
            'train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)'
        )
        source_str = source_str.replace(old_train_line, mixup_code)
        changes_log.append('Added Mixup augmentation')

    # ===== 5. CHECKPOINT FORMAT (weights-only for FocalLoss compatibility) =====

    source_str = source_str.replace(
        "filepath='best_multimodal_model.keras',",
        "filepath='best_multimodal_model.weights.h5', save_weights_only=True,")

    source_str = source_str.replace(
        "filepath='best_multimodal_model_ft.keras',",
        "filepath='best_multimodal_model_ft.weights.h5', save_weights_only=True,")

    source_str = source_str.replace(
        "model.load_weights('best_multimodal_model.keras')",
        "model.load_weights('best_multimodal_model.weights.h5')")

    # ===== 6. SAVE MODEL AS V3 + RECOMPILE FOR API COMPATIBILITY =====

    # Replace model save filename: multimodal_emotion_model.keras -> multimodal_emotion_model_v3.keras
    source_str = source_str.replace(
        'model.save(os.path.join(SAVE_DIR, "multimodal_emotion_model.keras"))',
        "# Recompile with standard loss so api.py can load without custom objects\n"
        "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n"
        'model.save(os.path.join(SAVE_DIR, "multimodal_emotion_model_v3.keras"))')

    # ===== TRACK CHANGES =====
    if source_str != original:
        cell['source'] = source_str.splitlines(True)
        # Ensure last line has newline
        if cell['source'] and not cell['source'][-1].endswith('\n'):
            cell['source'][-1] += '\n'

        if 'Conv2D(128' in source_str and 'Conv2D(128' not in original:
            changes_log.append('CNN filters widened 64->128')
        if 'Dense(128' in source_str and 'Dense(128' not in original:
            changes_log.append('Embeddings widened 64->128')
        if 'key_dim=128' in source_str and 'key_dim=64' in original:
            changes_log.append('Attention key_dim 64->128')
        if 'Dropout(0.4' in source_str and 'Dropout(0.5' in original:
            changes_log.append('Fusion dropout reduced 0.5->0.4')
        if '.batch(32)' in source_str and '.batch(16)' in original:
            changes_log.append('Batch size 16->32')
        if 'buffer_size=5000' in source_str and 'buffer_size=10000' in original:
            changes_log.append('Shuffle buffer 10000->5000')
        if 'epochs=50' in source_str and 'epochs=80' in original:
            changes_log.append('Phase 1 epochs 80->50')
        if 'epochs=25' in source_str and 'epochs=40' in original:
            changes_log.append('Phase 2 epochs 40->25')
        if 'learning_rate=5e-5' in source_str and 'learning_rate=2e-5' in original:
            changes_log.append('Phase 2 LR 2e-5->5e-5')
        if 'save_weights_only=True' in source_str and 'save_weights_only' not in original:
            changes_log.append('Checkpoints -> weights-only format')
        if 'multimodal_emotion_model_v3' in source_str and 'multimodal_emotion_model_v3' not in original:
            changes_log.append('Model save name -> v3')
        if "loss='categorical_crossentropy'" in source_str and "loss='categorical_crossentropy'" not in original:
            changes_log.append('Added recompile before save for API compatibility')

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("=" * 60)
print("ALL CHANGES APPLIED SUCCESSFULLY!")
print("=" * 60)
for change in changes_log:
    print(f"  [OK] {change}")
print(f"\nTotal: {len(changes_log)} modifications applied.")
print("\nIMPORTANT: After training, update api.py line 76:")
print('  OLD: model_path = os.path.join(MODEL_DIR, "multimodal_emotion_model.keras")')
print('  NEW: model_path = os.path.join(MODEL_DIR, "multimodal_emotion_model_v3.keras")')
