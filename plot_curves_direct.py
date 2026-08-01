import matplotlib.pyplot as plt

# Hardcoded data exactly from the training logs provided by the user
train_acc = [
    31.25, 42.32, 48.21, 52.30, 56.70, 59.53, 62.56, 64.75, 67.24, 69.02, 72.05, 73.94, 76.46, 78.33, 79.53, 81.09, 82.70, 83.59, # Phase 1
    72.33, 72.44, 73.16, 73.32, 73.94, 74.14, 73.59, 73.42, 74.10, 73.56, 74.37, 74.16, 73.91, 73.87,                             # Phase 2
    74.73, 73.83, 74.31, 74.98, 75.09, 74.92                                                                                      # Phase 3
]
val_acc = [
    46.50, 50.03, 51.39, 55.15, 56.86, 54.30, 57.71, 57.88, 59.08, 58.45, 53.04, 59.13, 58.45, 58.51, 57.43, 57.31, 56.86, 60.22, 
    59.82, 60.44, 60.44, 60.96, 60.22, 59.93, 60.44, 60.33, 60.27, 61.01, 60.22, 60.16, 61.01, 60.22, 
    59.65, 60.84, 59.93, 60.90, 60.96, 60.61
]

train_loss = [
    0.9178, 0.7984, 0.7394, 0.6870, 0.6356, 0.5961, 0.5561, 0.5201, 0.4919, 0.4602, 0.4222, 0.3986, 0.3806, 0.3562, 0.3300, 0.3139, 0.2955, 0.2842,
    0.4299, 0.4270, 0.4155, 0.4152, 0.4169, 0.4102, 0.4118, 0.4110, 0.4140, 0.4167, 0.4116, 0.4107, 0.4039, 0.4127,
    0.3912, 0.3880, 0.3852, 0.3830, 0.3831, 0.3842
]
val_loss = [
    0.7479, 0.7136, 0.6832, 0.6671, 0.6286, 0.6705, 0.6454, 0.6373, 0.6621, 0.6842, 0.7492, 0.6945, 0.7034, 0.6760, 0.7582, 0.7448, 0.8030, 0.7716,
    0.6818, 0.6522, 0.6782, 0.6617, 0.6765, 0.6993, 0.6748, 0.6719, 0.6696, 0.6843, 0.6715, 0.6772, 0.6717, 0.6741,
    0.6766, 0.6831, 0.6873, 0.6870, 0.6816, 0.6781
]

epochs = range(1, len(train_acc) + 1)
phase_2_start = 18  # End of Phase 1
phase_3_start = 32  # End of Phase 2

plt.figure(figsize=(14, 6))

# Subplot 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Train Accuracy', color='blue', linewidth=2)
plt.plot(epochs, val_acc, label='Validation Accuracy', color='orange', linewidth=2)
plt.axvline(x=phase_3_start, color='gray', linestyle='--', label='Start Phase 3')
plt.axvline(x=phase_2_start, color='black', linestyle='--', label='Start Phase 2')
plt.title('Training & Validation Accuracy (Up to Phase 3)')
plt.xlabel('Total Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Subplot 2: Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linewidth=2)
plt.axvline(x=phase_3_start, color='gray', linestyle='--', label='Start Phase 3')
plt.axvline(x=phase_2_start, color='black', linestyle='--', label='Start Phase 2')
plt.title('Training & Validation Loss (Up to Phase 3)')
plt.xlabel('Total Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('Phase3_Curves.png', dpi=300)
print("Graph saved successfully!")
