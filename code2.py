#Phone detection with alarm sound


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pygame

pygame.mixer.init() 

# Load YOLO model
config_path = "yolov4.cfg"
weights_path = "yolov4.weights"
labels_path = "coco.names"

# Load class labels
with open(labels_path, "r") as f:
    labels = f.read().strip().split("\n")

# Initialize YOLO
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Folder containing extracted frames
frames_folder = "frames1"
#frames_folder = "frames2"  # Path to the folder containing the frames

# Get list of frames in the folder
frame_files = [f for f in os.listdir(frames_folder) if f.endswith(".jpg") or f.endswith(".png")]

# Variables to track the best frame and detection status
best_frame = None
best_confidence = 0
frame_confidences = []
phone_detected = False  # Flag to check if a phone is detected

# Process each frame
for frame_file in frame_files:
    # Read the frame
    frame_path = os.path.join(frames_folder, frame_file)
    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    # Analyze detections
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter for mobile phones (class_id depends on `coco.names`)
            if labels[class_id] == "cell phone" and confidence > 0.5:
                phone_detected = True  # Phone is detected
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check if there's at least one detection
    if len(idxs) > 0:
        # Calculate the sum of confidences for the frame
        total_confidence = sum([confidences[i] for i in idxs.flatten()])
        frame_confidences.append(total_confidence)

        # Track the frame with the highest confidence
        if total_confidence > best_confidence:
            best_confidence = total_confidence
            best_frame = frame

# Print detection status
if phone_detected:
    print("Phone detected")
else:
    print("No phone detected")

# Plotting the confidence of each frame
if frame_confidences:
    plt.plot(frame_files, frame_confidences, marker='o')
    plt.title("Confidence of Object Detection Across Frames")
    plt.xlabel("Frame Number")
    plt.ylabel("Total Confidence")
    plt.xticks(rotation=90)
    plt.show()  # Show confidence graph and wait for closure

# Play alarm sound after the graph is closed
if phone_detected:
    pygame.mixer.music.load("motion_detected.mp3")  # Replace with your audio file path
    pygame.mixer.music.play()  # Play sound during image display

# If a best frame is found, display it with bounding boxes
if best_frame is not None:
    (H, W) = best_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(best_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if labels[class_id] == "cell phone" and confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes on the best frame
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            color = (0, 255, 0)  # Green for bounding boxes
            cv2.rectangle(best_frame, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(best_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the best frame with detected object
    plt.imshow(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()  # Show the image
