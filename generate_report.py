import json
import re
import matplotlib.pyplot as plt
import numpy as np

notebook_path = "main_multi_attention.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

outputs = []
for cell in nb['cells']:
    for out in cell.get('outputs', []):
        if out.get('name') == 'stdout':
            outputs.extend(out.get('text', []))

full_log = "".join(outputs)

# 1. Parse Training Curves
phases_train_acc = []
phases_val_acc = []
phases_train_loss = []
phases_val_loss = []

# Regex to find:
# Train Acc: 31.25% | Train Loss: 0.9178
# Val Acc:   46.50% | Val Loss:   0.7479
# We can just find all occurrences of Train Acc and Val Acc
train_matches = re.findall(r"Train Acc:\s*([\d.]+)%\s*\|\s*Train Loss:\s*([\d.]+)", full_log)
val_matches = re.findall(r"Val Acc:\s*([\d.]+)%\s*\|\s*Val Loss:\s*([\d.]+)", full_log)

epochs = range(1, len(train_matches) + 1)
train_acc = [float(m[0]) for m in train_matches]
train_loss = [float(m[1]) for m in train_matches]
val_acc = [float(m[0]) for m in val_matches]
val_loss = [float(m[1]) for m in val_matches]

plt.figure(figsize=(14, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Train Accuracy', color='blue', linewidth=2)
plt.plot(epochs, val_acc, label='Validation Accuracy', color='orange', linewidth=2)
plt.axvline(x=len(train_acc) - 20, color='gray', linestyle='--', label='Start Phase 3')
plt.axvline(x=len(train_acc) - 35, color='black', linestyle='--', label='Start Phase 2')
plt.title('Training & Validation Accuracy')
plt.xlabel('Total Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linewidth=2)
plt.axvline(x=len(train_loss) - 20, color='gray', linestyle='--', label='Start Phase 3')
plt.axvline(x=len(train_loss) - 35, color='black', linestyle='--', label='Start Phase 2')
plt.title('Training & Validation Loss')
plt.xlabel('Total Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
print("Saved training_curves.png")

# 2. Parse Classification Report
report_match = re.search(r"Classification Report:\n\s*precision\s*recall\s*f1-score\s*support\n(.*?)accuracy", full_log, re.DOTALL)
if report_match:
    lines = report_match.group(1).strip().split('\n')
    classes = []
    precision = []
    recall = []
    f1 = []
    
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            classes.append(parts[0])
            precision.append(float(parts[1]))
            recall.append(float(parts[2]))
            f1.append(float(parts[3]))
            
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, precision, width, label='Precision', color='#1f77b4')
    rects2 = ax.bar(x, recall, width, label='Recall', color='#ff7f0e')
    rects3 = ax.bar(x + width, f1, width, label='F1 Score', color='#2ca02c')

    ax.set_ylabel('Scores')
    ax.set_title('Precision, Recall, and F1 Score by Emotion Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim([0, 1.1])

    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig('classification_metrics.png', dpi=300)
    print("Saved classification_metrics.png")
else:
    print("Classification report not found in the notebook output. Ensure the final cell was run.")
