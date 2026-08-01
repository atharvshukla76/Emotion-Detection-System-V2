import matplotlib.pyplot as plt
import numpy as np

# Data derived from the user-provided Confusion Matrix (%)
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']

# True Class (rows) -> Predicted Class (columns)
# Values are in percentages (so rows sum to roughly 100)
C = np.array([
    [81.1, 4.2, 1.3, 11.1, 2.3, 0.0],
    [13.1, 50.2, 8.4, 11.4, 9.1, 7.7],
    [8.8, 5.1, 51.2, 12.8, 4.4, 17.8],
    [18.0, 4.7, 9.5, 59.3, 5.8, 2.7],
    [4.2, 4.6, 2.3, 8.4, 72.2, 8.4],
    [3.7, 9.4, 11.7, 5.4, 17.4, 52.3]
])

# Recall is simply the diagonal (since rows sum to 100%)
recall = np.diag(C) / 100.0

# Precision is the diagonal divided by the column sums
# (Assuming roughly equal support across classes for approximation)
col_sums = np.sum(C, axis=0)
precision = np.diag(C) / col_sums

# F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
f1 = 2 * (precision * recall) / (precision + recall)

# Plotting
x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, precision, width, label='Precision', color='#1f77b4')
rects2 = ax.bar(x, recall, width, label='Recall', color='#ff7f0e')
rects3 = ax.bar(x + width, f1, width, label='F1 Score', color='#2ca02c')

ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, and F1 Score by Emotion Class')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend(loc='lower right')
ax.set_ylim([0, 1.1])

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('classification_metrics.png', dpi=300)
print("Classification metrics graph saved.")
