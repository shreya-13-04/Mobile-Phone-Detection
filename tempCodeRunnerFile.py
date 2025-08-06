import cv2
import numpy as np
import os
import time
import queue
import threading
from playsound import playsound  
from ultralytics import YOLO  # ✅ New YOLOv8 Model

# ✅ Set paths
BASE_DIR = r"C:\Users\Shreya\Desktop\EvolveTag\phone_detection"
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")  # ✅ Use YOLOv8 model
ALARM_SOUND = os.path.join(BASE_DIR, "warning.mp3")

# ✅ Load YOLOv8 Model
model = YOLO(MODEL_PATH)  # Load pre-trained YOLOv8

# ✅ Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

if not cap.isOpened():
    print("❌ Error: Unable to open camera.")
    exit()

frame_queue = queue.Queue()
last_alert_time = 0  
frame_skip = 3  # Process every 3rd frame for speed

# ✅ Function to read frames in a separate thread
def read_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)

threading.Thread(target=read_frames, daemon=True).start()

# ✅ Main detection loop
try:
    frame_count = 0
    while True:
        if frame_queue.empty():
            continue  

        frame = frame_queue.get()
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  

        height, width = frame.shape[:2]
        roi = frame[height // 3:, :]  # ✅ Focus on hands/lap area for better detection

        # ✅ Run YOLOv8 on the frame
        results = model(roi, imgsz=640, conf=0.3)  # ✅ Lower confidence threshold for better detection

        phone_detected = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x, y, w, h = map(int, box.xywh[0])  

                # ✅ YOLOv8 class 67 = "cell phone" in COCO dataset
                if cls_id == 67 and confidence > 0.3:  
                    phone_detected = True

                    # ✅ Adjust bounding box to capture all edges
                    x, y, w, h = max(0, x - 10), max(0, y - 10), w + 20, h + 20  

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, f"📱 Phone Detected! {confidence:.2f}", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    print(f"🚨 Phone Detected! Confidence: {confidence:.2f}")

        # ✅ Trigger alarm (only once every 5 seconds)
        current_time = time.time()
        if phone_detected and (current_time - last_alert_time > 5):  
            print("🚨 Phone Detected! Triggering Alarm...")
            threading.Thread(target=playsound, args=(ALARM_SOUND,), daemon=True).start()  
            last_alert_time = current_time

        # ✅ Show video feed
        cv2.imshow("📹 Live Phone Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("❌ Stream interrupted manually.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera stream closed.")
