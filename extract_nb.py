import json

with open('main_multi_attention.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = []
for c in nb['cells']:
    if c['cell_type'] == 'code':
        code_cells.append(''.join(c['source']))

with open('extracted_code.py', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(code_cells))
