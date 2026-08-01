import json
import sys

try:
    with open('main_multi_attention.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for i, c in enumerate(nb['cells']):
        lines = c.get('source', [])
        preview = "".join(lines[:3]).strip()
        if len(preview) > 100:
            preview = preview[:100] + "..."
        print(f"Cell {i} ({c['cell_type']}) [len={len(lines)}]: {preview}")
except Exception as e:
    print(f"Error: {e}")
