import torch
import torch.nn as nn

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
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.1)
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
