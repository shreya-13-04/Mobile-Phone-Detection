import cv2
import matplotlib.pyplot as plt

# Load original image
image_path = 'phones_detected.png'
original = cv2.imread(image_path)

# Brightness enhancement (same as before)
hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v = cv2.add(v, 60)
final_hsv = cv2.merge((h, s, v))
brightened = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# Plot side-by-side comparison
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('After Brightness Enhancement')
plt.imshow(cv2.cvtColor(brightened, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

# Save the comparison figure if needed
plt.savefig('data_preprocessing_comparison.png')
