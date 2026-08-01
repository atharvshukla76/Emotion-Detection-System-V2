"""
apply_pipeline_fixes.py
Patches main_multi_attention.ipynb to fix:
  1. Actor leakage (train_test_split -> GroupShuffleSplit)
  2. Unused class_weights_tensor (wire into FocalLoss)
  3. Video reshape scrambles optical flow (fix 5D->4D permute)
  3b. Video augmentation indexing (add frame dimension)
  4. Zero-padded modalities not masked (add modality flags)
  Minor: Remove duplicate BATCH_SIZE, fix deprecated pretrained=True
"""
import json, sys

NOTEBOOK = "main_multi_attention.ipynb"

with open(NOTEBOOK, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def find_cell(marker):
    """Find a cell whose source contains the marker string."""
    for i, cell in enumerate(nb['cells']):
        src = "".join(cell.get('source', []))
        if marker in src:
            return i
    return -1

def replace_cell_source(idx, new_source_lines):
    """Replace a cell's source and clear its outputs."""
    nb['cells'][idx]['source'] = new_source_lines
    nb['cells'][idx]['outputs'] = []
    nb['cells'][idx]['execution_count'] = None

# ---------------------------------------------------------------
# FIX MINOR: Remove duplicate BATCH_SIZE from config cell
# ---------------------------------------------------------------
config_idx = find_cell("BATCH_SIZE = 128")
if config_idx >= 0:
    src = nb['cells'][config_idx]['source']
    new_src = [line for line in src if "BATCH_SIZE = 128" not in line]
    nb['cells'][config_idx]['source'] = new_src
    print(f"[OK] Removed duplicate BATCH_SIZE=128 from cell {config_idx}")
else:
    print("[SKIP] BATCH_SIZE=128 not found (already removed?)")

# ---------------------------------------------------------------
# FIX 1 + 3b + 4 (partial): Dataset compilation cell
# ---------------------------------------------------------------
dataset_idx = find_cell("MEMORY MAP DATASET")
if dataset_idx < 0:
    dataset_idx = find_cell("DATASET COMPILATION")

if dataset_idx >= 0:
    src_text = "".join(nb['cells'][dataset_idx].get('source', []))

    if "train_test_split" in src_text:
        replace_cell_source(dataset_idx, [
            "# =====================================================================\n",
            "# MEMORY MAP DATASET & ACTOR-GROUPED SPLIT (Bug Fix #1, #3b, #4)\n",
            "# =====================================================================\n",
            "from torch.utils.data import Dataset, DataLoader\n",
            "from sklearn.model_selection import GroupShuffleSplit\n",
            "\n",
            "BATCH_SIZE = 32\n",
            "\n",
            "class MMAPDataset(Dataset):\n",
            "    def __init__(self, aud, vid, y, augment=False):\n",
            "        self.aud = aud\n",
            "        self.vid = vid\n",
            "        self.y = y\n",
            "        self.augment = augment\n",
            "\n",
            "    def __len__(self):\n",
            "        return len(self.y)\n",
            "\n",
            "    def __getitem__(self, idx):\n",
            "        a = torch.tensor(self.aud[idx], dtype=torch.float32)\n",
            "        v = torch.tensor(self.vid[idx], dtype=torch.float32)\n",
            "        label = torch.tensor(self.y[idx], dtype=torch.long)\n",
            "\n",
            "        # Bug Fix #4: Modality-present flags\n",
            "        aud_present = 1.0 if a.abs().sum() > 0 else 0.0\n",
            "        vid_present = 1.0 if v.abs().sum() > 0 else 0.0\n",
            "        flags = torch.tensor([aud_present, vid_present], dtype=torch.float32)\n",
            "\n",
            "        if self.augment:\n",
            "            # Audio augmentation: time-mask a random band\n",
            "            if torch.rand(1).item() > 0.5:\n",
            "                t_mask = torch.randint(5, 30, (1,)).item()\n",
            "                t0 = torch.randint(0, 150 - t_mask, (1,)).item()\n",
            "                a[t0:t0+t_mask, :, :] = 0\n",
            "            # Video augmentation: spatial mask across ALL frames (Bug Fix #3b)\n",
            "            if torch.rand(1).item() > 0.5:\n",
            "                h_mask = torch.randint(10, 25, (1,)).item()\n",
            "                w_mask = torch.randint(10, 25, (1,)).item()\n",
            "                h0 = torch.randint(0, 64 - h_mask, (1,)).item()\n",
            "                w0 = torch.randint(0, 64 - w_mask, (1,)).item()\n",
            "                v[:, h0:h0+h_mask, w0:w0+w_mask, :] = 0  # v is (15, 64, 64, 2)\n",
            "\n",
            "        return a, v, label, flags\n",
            "\n",
            "# Bug Fix #1: Actor-grouped split - no actor in both train AND val\n",
            "gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)\n",
            "train_idx, val_idx = next(gss.split(np.arange(len(y_encoded)), y_encoded, groups=actors_all))\n",
            "\n",
            "train_dataset = MMAPDataset(X_audio_all[train_idx], X_video_all[train_idx], y_encoded[train_idx], augment=True)\n",
            "val_dataset = MMAPDataset(X_audio_all[val_idx], X_video_all[val_idx], y_encoded[val_idx], augment=False)\n",
            "\n",
            "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)\n",
            "val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)\n",
            "\n",
            "print(f\"Training Samples: {len(train_dataset)} | Validation Samples: {len(val_dataset)}\")\n",
        ])
        print(f"[OK] Fixed cell {dataset_idx}: GroupShuffleSplit + modality flags + video aug indexing")
    else:
        print(f"[WARN] Cell {dataset_idx} found but no train_test_split")
else:
    print("[ERROR] Could not find dataset/split cell!")
    sys.exit(1)


# ---------------------------------------------------------------
# FIX 3 + 4 (model) + MINOR (deprecated pretrained): Architecture cell
# ---------------------------------------------------------------
model_idx = find_cell("MOBILENET-V2 ENGINE SWAP")
if model_idx < 0:
    model_idx = find_cell("PYTORCH QUAD-MODAL ARCHITECTURE")

if model_idx >= 0:
    replace_cell_source(model_idx, [
        "# =====================================================================\n",
        "# PYTORCH QUAD-MODAL ARCHITECTURE (MOBILENETV2 - ALL BUGS FIXED)\n",
        "# =====================================================================\n",
        "import torchvision.models as models\n",
        "from torchvision.models import MobileNet_V2_Weights\n",
        "import torch.nn as nn\n",
        "import torch\n",
        "\n",
        "class AudioBranch(nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        self.mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)\n",
        "        self.channel_map = nn.Conv2d(1, 3, kernel_size=1)\n",
        "        self.mobilenet.classifier = nn.Identity()\n",
        "        self.dense = nn.Linear(1280, 128)\n",
        "        self.relu = nn.ReLU()\n",
        "        self.drop = nn.Dropout(0.3)\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = x.permute(0, 3, 1, 2)  # [batch, 1, 150, 136]\n",
        "        x = self.channel_map(x)     # [batch, 3, 150, 136]\n",
        "        x = self.mobilenet(x)       # [batch, 1280]\n",
        "        x = self.drop(x)\n",
        "        return self.relu(self.dense(x))\n",
        "\n",
        "class VideoBranch(nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        self.mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)\n",
        "        self.channel_map = nn.Conv2d(30, 3, kernel_size=1)\n",
        "        self.mobilenet.classifier = nn.Identity()\n",
        "        self.dense = nn.Linear(1280, 128)\n",
        "        self.relu = nn.ReLU()\n",
        "        self.drop = nn.Dropout(0.3)\n",
        "\n",
        "    def forward(self, x):\n",
        "        # Bug Fix #3: x is [batch, 15, 64, 64, 2]\n",
        "        # Move flow-channels (2) next to frames (15), then collapse to 30\n",
        "        x = x.permute(0, 1, 4, 2, 3).contiguous()  # [batch, 15, 2, 64, 64]\n",
        "        batch_size = x.size(0)\n",
        "        x = x.view(batch_size, 30, 64, 64)          # [batch, 30, 64, 64]\n",
        "        x = self.channel_map(x)                      # [batch, 3, 64, 64]\n",
        "        x = self.mobilenet(x)                        # [batch, 1280]\n",
        "        x = self.drop(x)\n",
        "        return self.relu(self.dense(x))\n",
        "\n",
        "class SEBlock(nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        self.fc1 = nn.Linear(128, 32)\n",
        "        self.fc2 = nn.Linear(32, 128)\n",
        "        self.relu = nn.ReLU()\n",
        "        self.sigmoid = nn.Sigmoid()\n",
        "\n",
        "    def forward(self, x):\n",
        "        sq = x.mean(dim=1)\n",
        "        ex = self.sigmoid(self.fc2(self.relu(self.fc1(sq)))).unsqueeze(1)\n",
        "        return x * ex\n",
        "\n",
        "class QuadModalModel(nn.Module):\n",
        "    def __init__(self, num_classes=6):\n",
        "        super().__init__()\n",
        "        self.audio = AudioBranch()\n",
        "        self.video = VideoBranch()\n",
        "        self.se = SEBlock()\n",
        "        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)\n",
        "\n",
        "        self.fc1 = nn.Linear(128 * 2, 128)\n",
        "        self.drop1 = nn.Dropout(0.6)\n",
        "        self.fc2 = nn.Linear(128, 64)\n",
        "        self.drop2 = nn.Dropout(0.3)\n",
        "        self.out = nn.Linear(64, num_classes)\n",
        "        self.relu = nn.ReLU()\n",
        "\n",
        "    def forward(self, aud, vid, modality_flags=None):\n",
        "        a = self.audio(aud).unsqueeze(1)  # [batch, 1, 128]\n",
        "        v = self.video(vid).unsqueeze(1)   # [batch, 1, 128]\n",
        "\n",
        "        # Bug Fix #4: Zero out absent modality embeddings\n",
        "        if modality_flags is not None:\n",
        "            a = a * modality_flags[:, 0:1].unsqueeze(-1)  # scale by aud_present\n",
        "            v = v * modality_flags[:, 1:2].unsqueeze(-1)  # scale by vid_present\n",
        "\n",
        "        x = self.se(torch.cat([a, v], dim=1))\n",
        "        attn_out, _ = self.attn(x, x, x)\n",
        "        x = x + attn_out\n",
        "        x = x.view(x.size(0), -1)\n",
        "        x = self.drop1(self.relu(self.fc1(x)))\n",
        "        x = self.drop2(self.relu(self.fc2(x)))\n",
        "        return self.out(x)\n",
        "\n",
        "model = QuadModalModel(num_classes=6).to(DEVICE)\n",
        "print(\"MobileNetV2 Engine Swap Complete! Model Ready!\")\n",
    ])
    print(f"[OK] Fixed cell {model_idx}: VideoBranch reshape + modality flags + deprecated API")
else:
    print("[ERROR] Could not find model architecture cell!")
    sys.exit(1)


# ---------------------------------------------------------------
# FIX 2: Phase 1 training loop
# ---------------------------------------------------------------
phase1_idx = find_cell("PHASE 1: PYTORCH EXTREME REGULARIZATION")
if phase1_idx >= 0:
    replace_cell_source(phase1_idx, [
        "# =====================================================================\n",
        "# PHASE 1: PYTORCH EXTREME REGULARIZATION TRAINING LOOP (MOBILENETV2)\n",
        "# =====================================================================\n",
        "import torch.optim as optim\n",
        "import time\n",
        "\n",
        "class FocalLoss(nn.Module):\n",
        "    def __init__(self, gamma=3.0, label_smoothing=0.15, weight=None):\n",
        "        super().__init__()\n",
        "        self.gamma = gamma\n",
        "        # Bug Fix #2: Pass class weights into CrossEntropyLoss\n",
        "        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction='none')\n",
        "    def forward(self, inputs, targets):\n",
        "        ce_loss = self.ce(inputs, targets)\n",
        "        pt = torch.exp(-ce_loss)\n",
        "        return (((1 - pt) ** self.gamma) * ce_loss).mean()\n",
        "\n",
        "criterion = FocalLoss(gamma=3.0, label_smoothing=0.15, weight=class_weights_tensor)\n",
        "\n",
        "optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-2)\n",
        "\n",
        "EPOCHS = 50\n",
        "scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)\n",
        "\n",
        "PATIENCE = 15\n",
        "best_val_acc = 0.0\n",
        "patience_counter = 0\n",
        "\n",
        "print(\"Starting MobileNetV2 Phase 1 (All Bugs Fixed)...\")\n",
        "for epoch in range(EPOCHS):\n",
        "    model.train()\n",
        "    train_loss, train_correct, train_total = 0, 0, 0\n",
        "    start_time = time.time()\n",
        "\n",
        "    for batch_idx, (a, v, y, flags) in enumerate(train_loader):\n",
        "        a, v, y, flags = a.to(DEVICE), v.to(DEVICE), y.to(DEVICE), flags.to(DEVICE)\n",
        "\n",
        "        optimizer.zero_grad()\n",
        "        outputs = model(a, v, flags)\n",
        "        loss = criterion(outputs, y)\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "\n",
        "        train_loss += loss.item()\n",
        "        _, predicted = outputs.max(1)\n",
        "        train_total += y.size(0)\n",
        "        train_correct += predicted.eq(y).sum().item()\n",
        "\n",
        "    model.eval()\n",
        "    val_loss, val_correct, val_total = 0, 0, 0\n",
        "    with torch.no_grad():\n",
        "        for a, v, y, flags in val_loader:\n",
        "            a, v, y, flags = a.to(DEVICE), v.to(DEVICE), y.to(DEVICE), flags.to(DEVICE)\n",
        "            outputs = model(a, v, flags)\n",
        "            loss = criterion(outputs, y)\n",
        "            val_loss += loss.item()\n",
        "            _, predicted = outputs.max(1)\n",
        "            val_total += y.size(0)\n",
        "            val_correct += predicted.eq(y).sum().item()\n",
        "\n",
        "    train_acc = 100. * train_correct / train_total\n",
        "    val_acc = 100. * val_correct / val_total\n",
        "    avg_val_loss = val_loss / len(val_loader)\n",
        "\n",
        "    scheduler.step()\n",
        "    epoch_time = time.time() - start_time\n",
        "    print(f\"\\nPhase 1 - Epoch {epoch+1} Summary ({epoch_time:.1f}s):\")\n",
        "    print(f\"Train Acc: {train_acc:.2f}% | Train Loss: {train_loss/len(train_loader):.4f}\")\n",
        "    print(f\"Val Acc:   {val_acc:.2f}% | Val Loss:   {avg_val_loss:.4f}\\n\")\n",
        "\n",
        "    if val_acc > best_val_acc:\n",
        "        print(f\"\\u2b50 Validation Accuracy improved to {val_acc:.2f}%. Saving model!\")\n",
        "        best_val_acc = val_acc\n",
        "        torch.save(model.state_dict(), 'best_pytorch_model.pt')\n",
        "        patience_counter = 0\n",
        "    else:\n",
        "        patience_counter += 1\n",
        "        if patience_counter >= PATIENCE:\n",
        "            print(f\"Early stopping triggered at epoch {epoch+1}\")\n",
        "            break\n",
    ])
    print(f"[OK] Fixed cell {phase1_idx}: FocalLoss weight + modality flags in loop")
else:
    print("[ERROR] Could not find Phase 1 training cell!")
    sys.exit(1)


# ---------------------------------------------------------------
# FIX 2 (Phase 2): Wire class_weights_tensor + unpack flags
# ---------------------------------------------------------------
phase2_idx = find_cell("PHASE 2: MOBILENETV2 FREEZE-TUNING")
if phase2_idx >= 0:
    replace_cell_source(phase2_idx, [
        "# =====================================================================\n",
        "# PHASE 2: MOBILENETV2 FREEZE-TUNING (ALL BUGS FIXED)\n",
        "# =====================================================================\n",
        "print(\"Loading the best Phase 1 MobileNetV2 model...\")\n",
        "model.load_state_dict(torch.load('best_pytorch_model.pt', map_location=DEVICE))\n",
        "\n",
        "# Freeze MobileNet backbones\n",
        "print(\"Freezing MobileNet backbones to lock in ImageNet knowledge...\")\n",
        "for param in model.audio.mobilenet.parameters():\n",
        "    param.requires_grad = False\n",
        "for param in model.video.mobilenet.parameters():\n",
        "    param.requires_grad = False\n",
        "\n",
        "model.drop1.p = 0.6\n",
        "model.drop2.p = 0.4\n",
        "\n",
        "criterion = FocalLoss(gamma=3.0, label_smoothing=0.15, weight=class_weights_tensor)\n",
        "optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-2)\n",
        "\n",
        "PHASE2_EPOCHS = 15\n",
        "scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-6)\n",
        "\n",
        "print(\"Starting Phase 2 Freeze-Tuning...\")\n",
        "best_val_acc_p2 = 0.0\n",
        "\n",
        "import time\n",
        "for epoch in range(PHASE2_EPOCHS):\n",
        "    model.train()\n",
        "    train_loss, train_correct, train_total = 0, 0, 0\n",
        "    start_time = time.time()\n",
        "\n",
        "    for batch_idx, (a, v, y, flags) in enumerate(train_loader):\n",
        "        a, v, y, flags = a.to(DEVICE), v.to(DEVICE), y.to(DEVICE), flags.to(DEVICE)\n",
        "\n",
        "        optimizer.zero_grad()\n",
        "        outputs = model(a, v, flags)\n",
        "        loss = criterion(outputs, y)\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "\n",
        "        train_loss += loss.item()\n",
        "        _, predicted = outputs.max(1)\n",
        "        train_total += y.size(0)\n",
        "        train_correct += predicted.eq(y).sum().item()\n",
        "\n",
        "    model.eval()\n",
        "    val_loss, val_correct, val_total = 0, 0, 0\n",
        "    with torch.no_grad():\n",
        "        for a, v, y, flags in val_loader:\n",
        "            a, v, y, flags = a.to(DEVICE), v.to(DEVICE), y.to(DEVICE), flags.to(DEVICE)\n",
        "            outputs = model(a, v, flags)\n",
        "            loss = criterion(outputs, y)\n",
        "            val_loss += loss.item()\n",
        "            _, predicted = outputs.max(1)\n",
        "            val_total += y.size(0)\n",
        "            val_correct += predicted.eq(y).sum().item()\n",
        "\n",
        "    train_acc = 100. * train_correct / train_total\n",
        "    val_acc = 100. * val_correct / val_total\n",
        "    avg_val_loss = val_loss / len(val_loader)\n",
        "\n",
        "    scheduler.step()\n",
        "    epoch_time = time.time() - start_time\n",
        "    print(f\"\\nPhase 2 - Epoch {epoch+1} Summary ({epoch_time:.1f}s):\")\n",
        "    print(f\"Train Acc: {train_acc:.2f}% | Train Loss: {train_loss/len(train_loader):.4f}\")\n",
        "    print(f\"Val Acc:   {val_acc:.2f}% | Val Loss:   {avg_val_loss:.4f}\\n\")\n",
        "\n",
        "    if val_acc > best_val_acc_p2:\n",
        "        print(f\"\\U0001f680 Validation Accuracy improved to {val_acc:.2f}%. Saving Phase 2 model!\")\n",
        "        best_val_acc_p2 = val_acc\n",
        "        torch.save(model.state_dict(), 'best_pytorch_model_phase2.pt')\n",
    ])
    print(f"[OK] Fixed cell {phase2_idx}: Phase 2 FocalLoss weight + modality flags")
else:
    print("[WARN] Phase 2 cell not found")


# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
with open(NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"\n{'='*60}")
print(f"All fixes applied to {NOTEBOOK}!")
print(f"{'='*60}")
