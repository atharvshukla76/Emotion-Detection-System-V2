import json

with open(r'd:\Emotion Detection system V2\main_multi_attention.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])

    if 'STEP 2: CROSS-MODAL' in src:
        print('=== ARCHITECTURE CELL ===')
        checks = [
            'Conv2D(128',
            'Dense(128',
            'Reshape((1, 128)',
            'key_dim=128',
            'Dropout(0.4',
        ]
        for c in checks:
            status = 'FOUND' if c in src else 'MISSING'
            print(f'  {c}: {status}')

    if 'class FocalLoss' in src:
        print('=== FOCAL LOSS ===')
        print(f'  FocalLoss class: FOUND')
        status = 'FOUND' if 'loss=FocalLoss' in src else 'MISSING'
        print(f'  loss=FocalLoss: {status}')

    if 'mixup_batch' in src:
        print('=== MIXUP ===')
        print(f'  mixup_batch function: FOUND')
        status = 'FOUND' if '.batch(32)' in src else 'MISSING'
        print(f'  batch(32): {status}')

    if 'multimodal_emotion_model_v3' in src:
        print('=== SAVE V3 ===')
        print(f'  v3 model name: FOUND')
        status = 'FOUND' if 'categorical_crossentropy' in src else 'MISSING'
        print(f'  recompile before save: {status}')

    if 'PHASE 2' in src and 'learning_rate' in src:
        print('=== PHASE 2 ===')
        status = 'FOUND' if '5e-5' in src else 'MISSING'
        print(f'  LR=5e-5: {status}')
        status = 'FOUND' if 'epochs=25' in src else 'MISSING'
        print(f'  epochs=25: {status}')
        status = 'FOUND' if 'weights.h5' in src else 'MISSING'
        print(f'  weights.h5: {status}')

print('\nVerification complete!')
