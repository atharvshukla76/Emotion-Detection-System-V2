import json

notebook_path = "main_multi_attention.ipynb"

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The missing dataset compilation and dataloader code
missing_cell_source = [
    "# =====================================================================\n",
    "# MEMORY MAP DATASET & ACTOR-GROUPED SPLIT (MISSING BLOCK RESTORED)\n",
    "# =====================================================================\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "\n",
    "class MMAPDataset(Dataset):\n",
    "    def __init__(self, aud, vid, y, augment=False):\n",
    "        self.aud = aud\n",
    "        self.vid = vid\n",
    "        self.y = y\n",
    "        self.augment = augment\n",
    "    \n",
    "    def __len__(self):\n",
    "        return len(self.y)\n",
    "    \n",
    "    def __getitem__(self, idx):\n",
    "        a = torch.tensor(self.aud[idx], dtype=torch.float32)\n",
    "        v = torch.tensor(self.vid[idx], dtype=torch.float32)\n",
    "        label = torch.tensor(self.y[idx], dtype=torch.long)\n",
    "        \n",
    "        aud_present = 1.0 if a.abs().sum() > 0 else 0.0\n",
    "        vid_present = 1.0 if v.abs().sum() > 0 else 0.0\n",
    "        flags = torch.tensor([aud_present, vid_present], dtype=torch.float32)\n",
    "        \n",
    "        if self.augment:\n",
    "            if aud_present and torch.rand(1).item() > 0.5:\n",
    "                a = a + torch.randn_like(a) * 0.05\n",
    "            if vid_present and torch.rand(1).item() > 0.5:\n",
    "                v = v + torch.randn_like(v) * 0.05\n",
    "                \n",
    "        return a, v, label, flags\n",
    "\n",
    "gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)\n",
    "train_idx, val_idx = next(gss.split(np.arange(len(y_encoded)), y_encoded, groups=actors_all))\n",
    "\n",
    "train_dataset = MMAPDataset(X_audio_all[train_idx], X_video_all[train_idx], y_encoded[train_idx], augment=True)\n",
    "val_dataset = MMAPDataset(X_audio_all[val_idx], X_video_all[val_idx], y_encoded[val_idx], augment=False)\n",
    "\n",
    "BATCH_SIZE = 32\n",
    "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)\n",
    "val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)\n",
    "\n",
    "print(f\"✅ Training Samples: {len(train_dataset)} | Validation Samples: {len(val_dataset)}\")\n"
]

missing_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": missing_cell_source
}

# Find index of Cell 8 (DATASET COMPILATION & LABEL ENCODING)
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and len(cell['source']) > 0 and 'DATASET COMPILATION & LABEL ENCODING' in cell['source'][1]:
        insert_idx = i + 1
        break

if insert_idx != -1:
    nb['cells'].insert(insert_idx, missing_cell)
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Successfully inserted missing DataLoader cell into notebook!")
else:
    print("Could not find insertion point!")
