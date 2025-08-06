import time
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
frame_files = [f for f in os.listdir(frames_folder) if f.endswith(".jpg") or f.endswith(".png")]

best_frame = None
best_confidence = 0
frame_confidences = []
phone_detected = False

# Start time for entire process
overall_start_time = time.time()

# Process each frame
for frame_file in frame_files:
    frame_path = os.path.join(frames_folder, frame_file)
    frame = cv2.imread(frame_path)
    
    if frame is None:
        continue

    start_time = time.time()  # Start time for processing this frame

    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass
    forward_start_time = time.time()
    layer_outputs = net.forward(output_layers)
    forward_end_time = time.time()

    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if labels[class_id] == "cell phone" and confidence > 0.5:
                phone_detected = True
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    nms_start_time = time.time()
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    nms_end_time = time.time()

    end_time = time.time()  # End time for processing this frame

    # Log timings
    print(f"Frame: {frame_file}")
    print(f"  Total Frame Processing Time: {end_time - start_time:.2f} seconds")
    print(f"  Forward Pass Time: {forward_end_time - forward_start_time:.2f} seconds")
    print(f"  NMS Time: {nms_end_time - nms_start_time:.2f} seconds")

    if len(idxs) > 0:
        total_confidence = sum([confidences[i] for i in idxs.flatten()])
        frame_confidences.append(total_confidence)
        if total_confidence > best_confidence:
            best_confidence = total_confidence
            best_frame = frame

# End time for entire process
overall_end_time = time.time()

# Log overall processing time
print(f"\nTotal Processing Time for All Frames: {overall_end_time - overall_start_time:.2f} seconds")

# Plot confidence graph
if frame_confidences:
    plt.plot(frame_files, frame_confidences, marker='o')
    plt.title("Confidence of Object Detection Across Frames")
    plt.xlabel("Frame Number")
    plt.ylabel("Total Confidence")
    plt.xticks(rotation=90)
    plt.show()

# Play alarm sound if a phone is detected
if phone_detected:
    pygame.mixer.music.load("motion_detected.mp3")
    pygame.mixer.music.play()
