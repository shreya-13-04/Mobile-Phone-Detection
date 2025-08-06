import cv2
import numpy as np
import os
import time
import queue
import threading
from playsound import playsound  

# Paths to model files and video source
BASE_DIR = r"C:\Users\Shreya\Desktop\EvolveTag\phone_detection"
LABELS_PATH = os.path.join(BASE_DIR, "coco.names")
CFG_PATH = os.path.join(BASE_DIR, "yolov4.cfg")
WEIGHTS_PATH = os.path.join(BASE_DIR, "yolov4.weights")
ALARM_SOUND = os.path.join(BASE_DIR, "warning.mp3")
VIDEO_SOURCE = os.path.join(BASE_DIR, "p6.mp4")  

# Load YOLO model
net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)

# Try using CUDA if available
try:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using CUDA for YOLO")
except:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU for YOLO")

# Get YOLO layers and labels
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open(LABELS_PATH, "r") as f:
    labels = f.read().strip().split("\n")

# Open video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: Unable to open video source.")
    exit()

frame_queue = queue.Queue()
last_alert_time = 0  # Last time an alert was triggered
frame_skip = 2  # Skip frames to optimize processing

# Function to read frames in a separate thread
def read_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)

threading.Thread(target=read_frames, daemon=True).start()

# Function to compute colorfulness
def compute_colorfulness(image):
    """Returns a colorfulness score for an image."""
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    mean_rg, std_rg = np.mean(rg), np.std(rg)
    mean_yb, std_yb = np.mean(yb), np.std(yb)
    return np.sqrt(std_rg ** 2 + std_yb ** 2) + (0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2))

try:
    frame_count = 0
    while True:
        if frame_queue.empty():
            continue  

        frame = frame_queue.get()
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip processing every nth frame

        height, width = frame.shape[:2]

        # Convert frame to blob and run YOLO
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        phone_detected = False

        # Analyze detections
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.3 and labels[class_id] == "cell phone":  # Process only phones
                    phone_detected = True

                    # Get bounding box
                    x, y, w, h = (obj[:4] * np.array([width, height, width, height])).astype("int")
                    x, y, w, h = max(0, x - w // 2), max(0, y - h // 2), w, h

                    # Extract ROI (Region of Interest)
                    roi = frame[y:y+h, x:x+w]

                    # Convert to grayscale for processing
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                    # Edge detection (higher sensitivity)
                    edges = cv2.Canny(gray, 30, 120)  # Adjusted threshold

                    # Contour detection
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Filter contours by area and shape
                    large_contours = [c for c in contours if cv2.contourArea(c) > 100]
                    num_large_contours = len(large_contours)

                    # Aspect Ratio (Width / Height)
                    aspect_ratio = w / float(h)

                    # Texture Analysis - Laplacian Variance
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

                    # Edge Density Calculation
                    edge_density = np.sum(edges) / (w * h)

                    # Compute Colorfulness Score
                    colorfulness = compute_colorfulness(roi)

                    # Decision logic: classify as phone or calculator
                    if (num_large_contours > 12 and aspect_ratio < 0.6 and edge_density > 0.05 
                        and laplacian_var > 150 and colorfulness < 20):  
                        label = "Calculator"
                        color = (0, 255, 255)  # Yellow
                    else:
                        label = "Phone"
                        color = (0, 0, 255)  # Red

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Alert when a phone is detected
        current_time = time.time()
        if phone_detected and (current_time - last_alert_time > 5):  # Avoid constant beeping
            print("Phone Detected! Triggering Alarm...")
            threading.Thread(target=playsound, args=(ALARM_SOUND,)).start()  # Non-blocking sound
            last_alert_time = current_time

        # Display live feed
        cv2.imshow("Live Surveillance Feed", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stream interrupted manually.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Video stream closed.")
