import json

notebook_path = "main_multi_attention.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update the exact cells we injected earlier
for cell in nb['cells']:
    source = "".join(cell.get('source', []))
    
    # 1. Update Dataset for Stronger Augmentations (SpecAugment & Spatial Erasing)
    if "class MMAPDataset(Dataset):" in source:
        new_source = []
        for line in cell['source']:
            if "if self.is_train:" in line:
                new_source.extend([
                    "            if self.is_train:\n",
                    "                # Audio Augmentation (Noise & Volume)\n",
                    "                if torch.rand(1).item() > 0.5:\n",
                    "                    a = a + torch.randn_like(a) * 0.05\n",
                    "                if torch.rand(1).item() > 0.5:\n",
                    "                    a = a * (0.8 + 0.4 * torch.rand(1).item())\n",
                    "                \n",
                    "                # AUDIO SPECAUGMENT (Time Masking)\n",
                    "                if torch.rand(1).item() > 0.5:\n",
                    "                    t_mask = torch.randint(5, 30, (1,)).item()\n",
                    "                    t0 = torch.randint(0, 150 - t_mask, (1,)).item()\n",
                    "                    a[t0:t0+t_mask, :, :] = 0\n",
                    "                \n",
                    "                # AUDIO SPECAUGMENT (Frequency Masking)\n",
                    "                if torch.rand(1).item() > 0.5:\n",
                    "                    f_mask = torch.randint(5, 30, (1,)).item()\n",
                    "                    f0 = torch.randint(0, 136 - f_mask, (1,)).item()\n",
                    "                    a[:, f0:f0+f_mask, :] = 0\n"
                ])
                # Skip the old augmentation lines
                continue
            
            if "a = a + torch.randn_like(a) * 0.05" in line or "a = a * (0.8 + 0.4 * torch.rand(1).item())" in line:
                continue
            
            if "v = (v - v.mean()) / (v.std() + 1e-6)" in line:
                new_source.append(line)
                new_source.extend([
                    "\n",
                    "            # VIDEO AUGMENTATION (Spatial Random Erasing)\n",
                    "            if self.is_train:\n",
                    "                if torch.rand(1).item() > 0.5:\n",
                    "                    h_mask = torch.randint(10, 25, (1,)).item()\n",
                    "                    w_mask = torch.randint(10, 25, (1,)).item()\n",
                    "                    h0 = torch.randint(0, 64 - h_mask, (1,)).item()\n",
                    "                    w0 = torch.randint(0, 64 - w_mask, (1,)).item()\n",
                    "                    v[h0:h0+h_mask, w0:w0+w_mask, :] = 0\n"
                ])
                continue

            new_source.append(line)
            
        cell['source'] = new_source

    # 2. Update QuadModalModel for Stronger Dropout
    if "class QuadModalModel(nn.Module):" in source:
        new_source = []
        for line in cell['source']:
            if "self.drop1 = nn.Dropout(" in line and "fc1" not in line: # For branches, don't change
                pass
            if "self.drop1 = nn.Dropout(" in line and "fc1" in new_source[-1]:
                new_source.append("        self.drop1 = nn.Dropout(0.6)  # INCREASED DROPOUT\n")
            elif "self.drop2 = nn.Dropout(" in line and "fc2" in new_source[-1]:
                new_source.append("        self.drop2 = nn.Dropout(0.4)  # INCREASED DROPOUT\n")
            else:
                new_source.append(line)
        cell['source'] = new_source
        
    # 3. Update Training Loop for Stronger Regularization
    if "optimizer = optim.AdamW(" in source or "criterion = FocalLoss(" in source:
        new_source = []
        for line in cell['source']:
            if "criterion = FocalLoss(" in line:
                new_source.append("criterion = FocalLoss(gamma=3.0, label_smoothing=0.15)  # STRONGER PENALTY\n")
            elif "optimizer = optim.AdamW(" in line:
                new_source.append("optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)  # LOWER LR, HIGHER L2 PENALTY\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with extreme regularization techniques (SpecAugment, Video Masking, High Dropout/L2)!")
