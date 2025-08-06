import cv2
import numpy as np
import os
import time
import queue
import threading
from playsound import playsound  

# Base directory
BASE_DIR = r"C:\Users\Shreya\Desktop\EvolveTag\phone_detection"
LABELS_PATH = os.path.join(BASE_DIR, "coco.names")
CFG_PATH = os.path.join(BASE_DIR, "yolov4.cfg")
WEIGHTS_PATH = os.path.join(BASE_DIR, "yolov4.weights")
ALARM_SOUND = os.path.join(BASE_DIR, "warning.mp3")

# Load YOLO model
net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)

try:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using CUDA for YOLO")
except:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU for YOLO")

# Get layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = f.read().strip().split("\n")

# Start webcam
cap = cv2.VideoCapture(0)  # Use laptop camera
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

frame_queue = queue.Queue()
last_alert_time = 0  
frame_skip = 2  # Skip frames for better performance

# Read frames in a separate thread
def read_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)

threading.Thread(target=read_frames, daemon=True).start()

try:
    frame_count = 0
    while True:
        if frame_queue.empty():
            continue  

        frame = frame_queue.get()
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        phone_detected = False
        height, width = frame.shape[:2]

        # Process detections
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and labels[class_id] == "cell phone":  
                    phone_detected = True
                    x, y, w, h = (obj[:4] * np.array([width, height, width, height])).astype("int")
                    x, y, w, h = max(0, x - w // 2), max(0, y - h // 2), w, h

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Phone Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Trigger alarm
        current_time = time.time()
        if phone_detected and (current_time - last_alert_time > 5):  
            print("Phone Detected! Triggering Alarm...")
            threading.Thread(target=playsound, args=(ALARM_SOUND,), daemon=True).start()  
            last_alert_time = current_time

        # Show video feed
        cv2.imshow("Live Phone Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stream interrupted manually.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera stream closed.")
