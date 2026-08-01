import json

with open('main_multi_attention.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']

# ============================================================
# PATCH the big save cell (cell 20): the one with SAVE_DIR + model.save
# - Recompile with standard loss before saving (so api.py loads without FocalLoss)
# - Rename saved file to multimodal_emotion_model_v2.keras
# ============================================================
target_old_save = 'model.save(os.path.join(SAVE_DIR, "multimodal_emotion_model.keras"))'
target_new_save = (
    "# Recompile with standard loss so api.py can load without needing FocalLoss\n"
    "print('Recompiling with standard loss for API compatibility...')\n"
    "model.compile(\n"
    "    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),\n"
    "    loss=tf.keras.losses.CategoricalCrossentropy(),\n"
    "    metrics=['accuracy']\n"
    ")\n"
    "model.save(os.path.join(SAVE_DIR, \"multimodal_emotion_model_v2.keras\"))"
)

target_old_print = 'print(f"\\n✅ Model saved successfully to {SAVE_DIR}/multimodal_emotion_model.keras")'
target_new_print = 'print(f"\\n✅ Model saved to {SAVE_DIR}/multimodal_emotion_model_v2.keras (recompiled with standard loss for api.py)")'

patched = False
for i, cell in enumerate(code_cells):
    src = ''.join(cell['source'])
    if 'SAVE_DIR' in src and 'model.save' in src and 'norm.pkl' in src and 'multimodal_emotion_model.keras' in src:
        src = src.replace(target_old_save, target_new_save)
        src = src.replace(target_old_print, target_new_print)
        # Also fix the t-SNE layer reference: fc_fusion_2 → fc_fusion_3 (new deeper head)
        src = src.replace(
            'model.get_layer("fc_fusion_2").output',
            'model.get_layer("fc_fusion_3").output'
        )
        cell['source'] = [src]
        print(f'Save cell (index {i}) patched: recompile + rename + tsne layer fix')
        patched = True
        break

if not patched:
    print('WARNING: Save cell not found!')

# ============================================================
# PATCH the load-and-test cell (cell 21): update model filename
# ============================================================
for i, cell in enumerate(code_cells):
    src = ''.join(cell['source'])
    if 'multimodal_emotion_model.keras' in src and 'load_model' in src and 'attn_score_model' in src:
        src = src.replace(
            'tf.keras.models.load_model("saved_model/multimodal_emotion_model.keras")',
            'tf.keras.models.load_model("saved_model/multimodal_emotion_model_v2.keras")'
        )
        cell['source'] = [src]
        print(f'Load cell (index {i}) patched: filename updated')
        break

# Write back
with open('main_multi_attention.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Notebook saved.')
