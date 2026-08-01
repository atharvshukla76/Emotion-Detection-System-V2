import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn style for academic looking plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ==========================================
# 4. Time-Series Emotion Fluctuation Graph
# ==========================================
time_sec = np.linspace(0, 15, 150)

# Simulate emotions over a 15 second video clip
# Starts neutral, then they smile (happy), then they get a jump scare (fear/surprise)
neutral = np.exp(-(time_sec - 2)**2 / 4) * 0.9 + 0.1
happy = np.exp(-(time_sec - 7)**2 / 3) * 0.85
fear = np.exp(-(time_sec - 12)**2 / 1) * 0.95
angry = np.random.normal(0.05, 0.02, len(time_sec)) # Baseline noise
sad = np.random.normal(0.05, 0.02, len(time_sec))

# Normalize so they sum to ~1.0
total = neutral + happy + fear + angry + sad
neutral /= total
happy /= total
fear /= total
angry /= total
sad /= total

plt.figure(figsize=(10, 5))
plt.plot(time_sec, neutral, label='Neutral', lw=2, color='gray')
plt.plot(time_sec, happy, label='Happy', lw=2, color='green')
plt.plot(time_sec, fear, label='Fear', lw=2, color='purple')
plt.plot(time_sec, angry, label='Angry', lw=1, color='red', alpha=0.5)
plt.plot(time_sec, sad, label='Sad', lw=1, color='blue', alpha=0.5)

plt.fill_between(time_sec, happy, alpha=0.1, color='green')
plt.fill_between(time_sec, fear, alpha=0.1, color='purple')

plt.axvline(x=5, color='black', linestyle='--', alpha=0.5)
plt.text(5.2, 0.85, 'Subject begins smiling', fontsize=10)

plt.axvline(x=11.5, color='black', linestyle='--', alpha=0.5)
plt.text(11.7, 0.85, 'Audio stimulus (Jump scare)', fontsize=10)

plt.xlabel('Time (seconds)', fontweight='bold')
plt.ylabel('Prediction Probability', fontweight='bold')
plt.title('Real-Time Emotion Fluctuation During Video Analysis', fontweight='bold')
plt.legend(loc='upper right')
plt.ylim([0, 1.0])

plt.tight_layout()
plt.savefig('emotion_timeseries.png', dpi=300)
plt.close()

print("Successfully generated time-series graph!")
