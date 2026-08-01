import json

notebook_path = "main_multi_attention.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Clean Cell 0 (Imports)
cell_0 = nb['cells'][0]['source']
new_cell_0 = []
for line in cell_0:
    if "tensorflow" in line or "keras" in line or "TF_" in line:
        continue
    new_cell_0.append(line)
nb['cells'][0]['source'] = new_cell_0

# 2. Clean Cell 4 (Config / Seeds)
cell_4 = nb['cells'][4]['source']
new_cell_4 = []
for line in cell_4:
    if "tf.random.set_seed" in line:
        new_cell_4.append("import torch\ntorch.manual_seed(42)\n")
    else:
        new_cell_4.append(line)
nb['cells'][4]['source'] = new_cell_4

# 3. Delete Cells 8, 9, 10
# We need to find the cells by matching strings in their source to be totally safe
to_delete = []
for i, cell in enumerate(nb['cells']):
    source_str = "".join(cell.get('source', []))
    if "CALL LOADERS + CREATE TRAIN/VAL" in source_str:
        to_delete.append(i)
    elif "MEMORY-EFFICIENT IN-PLACE NORMALIZATION" in source_str:
        to_delete.append(i)
    elif "PRE-RESHAPE VIDEO ARRAYS" in source_str:
        to_delete.append(i)

# Delete in reverse order to not mess up indices
for i in sorted(to_delete, reverse=True):
    del nb['cells'][i]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Notebook cleaned! Removed TensorFlow and deleted {len(to_delete)} heavy RAM cells.")
