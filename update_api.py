import os

def update_api():
    api_path = "api.py"
    with open(api_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update imports
    content = content.replace("import tensorflow as tf", "import torch\nimport torch.nn as nn")

    # 2. Add PyTorch Model definitions
    pytorch_model_code = """
# =====================================================================
# PYTORCH QUAD-MODAL ARCHITECTURE
# =====================================================================
class AudioBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(0.3)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv1d_1 = nn.Conv1d(2176, 64, 1, padding=0)
        self.bn5 = nn.BatchNorm1d(64)
        self.drop4 = nn.Dropout(0.3)
        self.conv1d_2 = nn.Conv1d(64, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm1d(32)
        self.dense = nn.Linear(32, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.drop1(self.pool1(self.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.relu(self.bn3(self.conv3(x)))))
        x = self.relu(self.bn4(self.conv4(x)))
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H, W * C)
        x = x.permute(0, 2, 1)
        x = self.drop4(self.relu(self.bn5(self.conv1d_1(x))))
        x = self.relu(self.bn6(self.conv1d_2(x)))
        x = x.mean(dim=2)
        x = self.relu(self.dense(x))
        return x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class VideoBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SeparableConv2d(30, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)
        self.conv2 = SeparableConv2d(32, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.35)
        self.conv3 = SeparableConv2d(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout(0.4)
        self.dense = nn.Linear(128, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.drop1(self.pool1(self.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.relu(self.bn3(self.conv3(x))))
        x = x.mean(dim=[2, 3])
        x = self.relu(self.dense(x))
        return x

class SEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        sq = x.mean(dim=1)
        ex = self.relu(self.fc1(sq))
        ex = self.sigmoid(self.fc2(ex)).unsqueeze(1)
        return x * ex

class QuadModalModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.audio = AudioBranch()
        self.video = VideoBranch()
        self.se = SEBlock()
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(128 * 2, 128)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.2)
        self.out = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, aud, vid):
        a = self.audio(aud).unsqueeze(1)
        v = self.video(vid).unsqueeze(1)
        x = torch.cat([a, v], dim=1)
        x = self.se(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x.view(x.size(0), -1)
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        return self.out(x)

# ──────────────────────────────────────────────
# 2. STARTUP — LOAD ALL MODELS
# ──────────────────────────────────────────────
"""
    content = content.replace("# ──────────────────────────────────────────────\n# 2. STARTUP — LOAD ALL MODELS\n# ──────────────────────────────────────────────", pytorch_model_code)

    # 3. Replace AV model loading logic
    old_startup = '''    # ── AV Model ──
    # Try V2 first, fall back to older model
    for candidate in [
        "multimodal_emotion_model_v2.keras",
        "multimodal_emotion_model.keras",
        "emotion_model.keras",
    ]:
        model_path = os.path.join(MODEL_DIR, candidate)
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"[STARTUP] AV model loaded: {candidate}")
            break
    if model is None:
        print("[WARNING] No AV model found — prediction will use FER+Text only.")'''

    new_startup = '''    # ── AV Model ──
    try:
        model = QuadModalModel(num_classes=6)
        # Look for the new PyTorch model
        model_path = "best_pytorch_model.pt"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            print(f"[STARTUP] PyTorch AV model loaded: {model_path}")
        else:
            model = None
            print("[WARNING] No PyTorch AV model found — prediction will use FER+Text only.")
    except Exception as e:
        model = None
        print(f"[WARNING] Failed to load PyTorch model: {e}")'''
        
    content = content.replace(old_startup, new_startup)

    # 4. Replace Inference logic
    old_inference = '''        # ── AV model inference ──
        n_classes = len(encoder.classes_)
        if model is not None:
            probs_av = model.predict(
                {"audio_input": aud_input, "video_input": vid_feat},
                verbose=0
            )[0]
        else:
            probs_av = np.ones(n_classes, dtype=np.float32) / n_classes  # Uniform fallback'''

    new_inference = '''        # ── AV model inference ──
        n_classes = len(encoder.classes_)
        if model is not None:
            a_tensor = torch.from_numpy(aud_input).float()
            v_tensor = torch.from_numpy(vid_feat).float()
            with torch.no_grad():
                outputs = model(a_tensor, v_tensor)
                probs_av = torch.softmax(outputs, dim=1).numpy()[0]
        else:
            probs_av = np.ones(n_classes, dtype=np.float32) / n_classes  # Uniform fallback'''

    content = content.replace(old_inference, new_inference)

    with open(api_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("api.py updated successfully.")

if __name__ == "__main__":
    update_api()
