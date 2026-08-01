import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

print("Generating Audio Spectrogram...")

# IMPORTANT: Change this path to point to a real WAV file in your AudioWAV folder!
audio_path = '../AudioWAV/1001_DFA_HAP_XX.wav'

# Load the audio file
y, sr = librosa.load(audio_path, sr=22050)

# Compute the Mel-Spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

# Plot the Spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-Spectrogram of Happy Emotion')
plt.tight_layout()

# Save the image
plt.savefig('research_paper/spectrogram_example.png', dpi=300)
plt.close()
print("Saved spectrogram_example.png")