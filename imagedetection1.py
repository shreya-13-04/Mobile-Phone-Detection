# Install required libraries (if not already installed)
# pip install ultralytics opencv-python matplotlib

from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load your local YOLOv8x model
model = YOLO('yolov8x.pt')  # (your local file)

# Step 2: Load the input image
image_path = 'phone_detection11.jpg'  # <-- Put your image file name here
image = cv2.imread(image_path)

# Step 3: Brightness Enhancement (important for dark classrooms)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v = cv2.add(v, 60)  # Increase brightness (+60 units)
final_hsv = cv2.merge((h, s, v))
bright_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# Step 4: Phone Detection
results = model.predict(source=bright_image, conf=0.3)  # Lower confidence to catch smaller phones

# Step 5: Draw red boxes around detected phones
for result in results:
    boxes = result.boxes.xyxy
    labels = result.boxes.cls
    scores = result.boxes.conf

    for box, label, score in zip(boxes, labels, scores):
        if int(label) == 67:  # Class 67 = Cell Phone in COCO dataset
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(image, f'Phone {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# Step 6: Display and Save Output
plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Save output image
cv2.imwrite('phones_detected_final.png', image)
