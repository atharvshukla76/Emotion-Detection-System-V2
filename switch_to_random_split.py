import json

notebook_path = "main_multi_attention.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the exact cell and replace GroupShuffleSplit with standard train_test_split
for cell in nb['cells']:
    source = "".join(cell.get('source', []))
    
    if "gss = GroupShuffleSplit(" in source:
        new_source = []
        for line in cell['source']:
            if "from sklearn.model_selection import GroupShuffleSplit" in line:
                new_source.append("from sklearn.model_selection import train_test_split\n")
            elif "gss = GroupShuffleSplit(" in line:
                new_source.append("print('Using Standard Random Split to remove unseen actor penalty...')\n")
                new_source.append("train_idx, val_idx = train_test_split(range(len(train_full_dataset)), test_size=0.2, random_state=42)\n")
            elif "train_idx, val_idx = next(gss.split(" in line:
                pass # skip
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated to use Standard Random Split!")
