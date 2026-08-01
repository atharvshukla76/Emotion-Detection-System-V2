import json, os

notebook_path = r'd:\Emotion Detection system V2\main_multi_attention.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'v2_l.trainable = False' in src:
        new_src = src.replace('v2_l.trainable = False', 'v2_l.trainable = True')
        cell['source'] = new_src.splitlines(True)
        print(f"Cell {idx} updated: Audio branch is now trainable for 7-class learning!")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated!")
