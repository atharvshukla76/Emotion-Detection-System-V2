import json

notebook_path = "main_multi_attention.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_arch = """# =====================================================================
# PYTORCH QUAD-MODAL ARCHITECTURE (MOBILENET-V2 ENGINE SWAP)
# =====================================================================
import torchvision.models as models
import torch.nn as nn
import torch

class AudioBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Load Pre-Trained MobileNetV2 (3.4M Parameters)
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        # Map 1-channel spectrogram to 3-channel
        self.channel_map = nn.Conv2d(1, 3, kernel_size=1)
        # Remove the classification head (we just want the raw features)
        self.mobilenet.classifier = nn.Identity()
        
        # MobileNetV2 outputs 1280 features
        self.dense = nn.Linear(1280, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # [batch, 1, 150, 136]
        x = self.channel_map(x)   # [batch, 3, 150, 136]
        x = self.mobilenet(x)     # [batch, 1280]
        x = self.drop(x)
        return self.relu(self.dense(x))

class VideoBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        # Compress 30 temporal frames down to 3 channels seamlessly
        self.channel_map = nn.Conv2d(30, 3, kernel_size=1)
        self.mobilenet.classifier = nn.Identity()
        
        # MobileNetV2 outputs 1280 features
        self.dense = nn.Linear(1280, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # [batch, 30, 64, 64]
        x = self.channel_map(x)   # [batch, 3, 64, 64]
        x = self.mobilenet(x)     # [batch, 1280]
        x = self.drop(x)
        return self.relu(self.dense(x))

class SEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 32); self.fc2 = nn.Linear(32, 128)
        self.relu = nn.ReLU(); self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        sq = x.mean(dim=1)
        ex = self.sigmoid(self.fc2(self.relu(self.fc1(sq)))).unsqueeze(1)
        return x * ex

class QuadModalModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.audio = AudioBranch()
        self.video = VideoBranch()
        self.se = SEBlock()
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        self.fc1 = nn.Linear(128 * 2, 128)
        # INCREASED DROPOUT TO 0.6
        self.drop1 = nn.Dropout(0.6) 
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.3)
        self.out = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, aud, vid):
        a = self.audio(aud).unsqueeze(1)
        v = self.video(vid).unsqueeze(1)
        x = self.se(torch.cat([a, v], dim=1))
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x.view(x.size(0), -1)
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        return self.out(x)

# Note: DEVICE must be defined earlier in the notebook
try:
    model = QuadModalModel(num_classes=6).to(DEVICE)
    print("MobileNetV2 Engine Swap Complete! Model Ready!")
except NameError:
    print("WARNING: DEVICE is not defined. Please define DEVICE = 'cpu' or 'cuda' before running this.")
"""

new_phase2 = """# =====================================================================
# PHASE 2: MOBILENETV2 FREEZE-TUNING
# =====================================================================
print("Loading the best Phase 1 MobileNetV2 model...")
model.load_state_dict(torch.load('best_pytorch_model.pt', map_location=DEVICE))

# 1. FREEZE THE MOBILENET BACKBONES!
print("Freezing MobileNet backbones to lock in ImageNet knowledge...")
for param in model.audio.mobilenet.parameters():
    param.requires_grad = False
for param in model.video.mobilenet.parameters():
    param.requires_grad = False

# We only want to train the final Dense layers and the Attention block now!
model.drop1.p = 0.6
model.drop2.p = 0.4

# 3. Microscopic Learning Rate
criterion = FocalLoss(gamma=3.0, label_smoothing=0.15)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-2)

PHASE2_EPOCHS = 15
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-6)

print("Starting Phase 2 Freeze-Tuning to punch through 70%...")
# We set this high so it only saves if it beats your Phase 1 High Score!
best_val_acc_p2 = 64.71

import time
for epoch in range(PHASE2_EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    start_time = time.time()
    
    for batch_idx, (a, v, y) in enumerate(train_loader):
        a, v, y = a.to(DEVICE), v.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(a, v)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += y.size(0)
        train_correct += predicted.eq(y).sum().item()
            
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for a, v, y in val_loader:
            a, v, y = a.to(DEVICE), v.to(DEVICE), y.to(DEVICE)
            outputs = model(a, v)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += y.size(0)
            val_correct += predicted.eq(y).sum().item()
            
    train_acc = 100. * train_correct / train_total
    val_acc = 100. * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    
    scheduler.step()
    epoch_time = time.time() - start_time
    print(f"\\nPhase 2 - Epoch {epoch+1} Summary ({epoch_time:.1f}s):")
    print(f"Train Acc: {train_acc:.2f}% | Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Val Acc:   {val_acc:.2f}% | Val Loss:   {avg_val_loss:.4f}\\n")
    
    if val_acc > best_val_acc_p2:
        print(f"🚀 Validation Accuracy skyrocketed to {val_acc:.2f}%. Saving Phase 2 model!")
        best_val_acc_p2 = val_acc
        torch.save(model.state_dict(), 'best_pytorch_model_phase2.pt')
"""

phase2_replaced = False
phase1_idx = -1

# Find Phase 1 block and Phase 2 block
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        
        if "PYTORCH EXTREME REGULARIZATION TRAINING LOOP (MOBILENETV2)" in source:
            phase1_idx = i
            
        if "PHASE 2: RESNET FREEZE-TUNING" in source:
            cell['source'] = [line + '\\n' for line in new_phase2.split('\\n')]
            cell['source'][-1] = cell['source'][-1].strip()
            phase2_replaced = True

# Inject architecture cell right before Phase 1 cell
if phase1_idx != -1:
    arch_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\\n' for line in new_arch.split('\\n')]
    }
    arch_cell['source'][-1] = arch_cell['source'][-1].strip()
    nb['cells'].insert(phase1_idx, arch_cell)
    print(f"Architecture injected before cell {phase1_idx}")
else:
    print("Could not find Phase 1 block to inject architecture before!")

print(f"Phase 2 replaced: {phase2_replaced}")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
