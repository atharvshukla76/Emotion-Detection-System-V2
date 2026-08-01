import json

notebook_path = "main_multi_attention.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_arch = """# =====================================================================
# PYTORCH QUAD-MODAL ARCHITECTURE (RESNET-18 ENGINE SWAP)
# =====================================================================
import torchvision.models as models

class AudioBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Load Pre-Trained ResNet-18 (The "Engine")
        # Note: pretrained=True downloads the weights automatically
        self.resnet = models.resnet18(pretrained=True)
        # Map 1-channel spectrogram to 3-channel for ResNet
        self.channel_map = nn.Conv2d(1, 3, kernel_size=1)
        # Remove the classification head (we just want the raw features)
        self.resnet.fc = nn.Identity()
        
        self.dense = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # [batch, 1, 150, 136]
        x = self.channel_map(x)   # [batch, 3, 150, 136]
        x = self.resnet(x)        # [batch, 512]
        x = self.drop(x)
        return self.relu(self.dense(x))

class VideoBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Compress 30 temporal frames down to 3 channels seamlessly
        self.channel_map = nn.Conv2d(30, 3, kernel_size=1)
        self.resnet.fc = nn.Identity()
        
        self.dense = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # [batch, 30, 64, 64]
        x = self.channel_map(x)   # [batch, 3, 64, 64]
        x = self.resnet(x)        # [batch, 512]
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
        self.drop1 = nn.Dropout(0.5)  # Slightly reduced since ResNet is already robust
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
print("ResNet-18 Engine Swap Complete! Model Ready!")
"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if "class AudioBranch(nn.Module):" in source and "class SeparableConv2d(nn.Module):" in source:
            start_marker = "# =====================================================================\n# PYTORCH QUAD-MODAL ARCHITECTURE"
            end_marker = 'print("PyTorch Model Architecture Ready!")\n'
            
            if start_marker in source and end_marker in source:
                before = source.split(start_marker)[0]
                after = source.split(end_marker)[1]
                
                final_source = before + new_arch + after
                
                cell['source'] = [line + '\n' for line in final_source.split('\n')]
                cell['source'] = [s.replace('\n\n', '\n') for s in cell['source']]
                print("Architecture successfully swapped in notebook!")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
