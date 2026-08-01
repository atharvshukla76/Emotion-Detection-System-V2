import matplotlib.pyplot as plt
import cv2

print("Generating Face Emotion Grid...")

# I updated these paths to match the beautiful new images in your folder!
image_paths = [
    'face_happy.jpg',
    'face_sad.jpg',
    'face_angry.jpg',
    'face_neutral.jpg'
]
emotion_labels = ['Happy', 'Sad', 'Angry', 'Neutral']

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, ax in enumerate(axes):
    try:
        # Read image and convert from BGR to RGB (so colors look normal)
        img = cv2.imread(image_paths[i])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img_rgb)
        ax.set_title(emotion_labels[i], fontsize=14, fontweight='bold')
        ax.axis('off') # Hides the axis numbers for a cleaner look
    except Exception as e:
        ax.set_title("Image Not Found")
        ax.axis('off')

plt.tight_layout()
plt.savefig('emotion_faces_grid.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved emotion_faces_grid.png")
