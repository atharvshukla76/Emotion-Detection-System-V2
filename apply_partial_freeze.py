import json

notebook_path = "main_multi_attention.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_arch = """# =====================================================================
# PYTORCH QUAD-MODAL ARCHITECTURE (PARTIAL FREEZE)
# =====================================================================
import torchvision.models as models

class AudioBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.channel_map = nn.Conv2d(1, 3, kernel_size=1)
        self.resnet.fc = nn.Identity()
        
        # FREEZE LAYERS 1, 2, 3 (The Overfitting Killer)
        for name, param in self.resnet.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
                
        self.dense = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.channel_map(x)
        x = self.resnet(x)
        x = self.drop(x)
        return self.relu(self.dense(x))

class VideoBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.channel_map = nn.Conv2d(30, 3, kernel_size=1)
        self.resnet.fc = nn.Identity()
        
        # FREEZE LAYERS 1, 2, 3 (The Overfitting Killer)
        for name, param in self.resnet.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
                
        self.dense = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.channel_map(x)
        x = self.resnet(x)
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
        self.drop1 = nn.Dropout(0.5) 
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

model = QuadModalModel(num_classes=6).to(DEVICE)
print("ResNet-18 Partial Freeze Complete! Model Ready!")
"""

new_phase1 = """# =====================================================================
# PYTORCH EXTREME REGULARIZATION TRAINING LOOP (PARTIAL FREEZE)
# =====================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=3.0, label_smoothing=0.15): 
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()

criterion = FocalLoss(gamma=3.0, label_smoothing=0.15)

# CRITICAL CHANGE: We only pass the UN-FROZEN parameters to the optimizer!
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-2) 

# CRITICAL CHANGE: Shortened to 30 Epochs because the model will learn much faster now
EPOCHS = 30
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

PATIENCE = 15
best_val_acc = 0.0
patience_counter = 0

import time
print("Starting Heavily Regularized PyTorch Training (Partial Freeze)...")
for epoch in range(EPOCHS):
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
    print(f"\\nEpoch {epoch+1} Summary ({epoch_time:.1f}s):")
    print(f"Train Acc: {train_acc:.2f}% | Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Val Acc:   {val_acc:.2f}% | Val Loss:   {avg_val_loss:.4f}\\n")
    
    if val_acc > best_val_acc:
        print(f"⭐ Validation Accuracy improved to {val_acc:.2f}%. Saving model!")
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_pytorch_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
"""

arch_replaced = False
phase1_replaced = False

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        
        # Replace Architecture Block
        if "class AudioBranch(nn.Module):" in source and "class SeparableConv2d(nn.Module):" not in source:
            cell['source'] = [line + '\\n' for line in new_arch.split('\\n')]
            cell['source'][-1] = cell['source'][-1].strip() # remove trailing newline
            arch_replaced = True
            
        # Replace Phase 1 Block
        if "class FocalLoss(nn.Module):" in source and "optimizer = optim.AdamW" in source:
            cell['source'] = [line + '\\n' for line in new_phase1.split('\\n')]
            cell['source'][-1] = cell['source'][-1].strip() # remove trailing newline
            phase1_replaced = True

print(f"Architecture replaced: {arch_replaced}")
print(f"Phase 1 replaced: {phase1_replaced}")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
