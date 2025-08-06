import cv2
import numpy as np
import os
import time
import queue
import threading
from playsound import playsound  
from ultralytics import YOLO  

# ✅ Set paths
BASE_DIR = r"C:\Users\Shreya\Desktop\EvolveTag\phone_detection"
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")  
ALARM_SOUND = os.path.join(BASE_DIR, "warning.mp3")

# ✅ Load YOLOv8 Model
model = YOLO(MODEL_PATH)  

# ✅ Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  
cap.set(cv2.CAP_PROP_FPS, 30)  # ✅ Ensure smooth video feed

if not cap.isOpened():
    print("❌ Error: Unable to open camera.")
    exit()

frame_queue = queue.Queue(maxsize=3)  # ✅ Limit queue size for better performance
last_alert_time = 0  
frame_skip = 3  # ✅ Process every 3rd frame for efficiency

# ✅ Function to read frames in a separate thread
def read_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
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
        roi = frame[height // 3:, :]  # ✅ Focus on hands/lap area

        # ✅ Run YOLOv8 on the frame
        results = model(roi, imgsz=640, conf=0.3)  

        phone_detected = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # ✅ Only detect cell phones (COCO class 67)
                if cls_id == 67 and confidence > 0.3:  
                    phone_detected = True
                    x, y, w, h = map(int, box.xywh[0])  

                    # ✅ Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, f"📱 Phone {confidence:.2f}", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    print(f"🚨 Phone Detected! Confidence: {confidence:.2f}")

        # ✅ Trigger alarm (only once every 5 seconds)
        current_time = time.time()
        if phone_detected and (current_time - last_alert_time > 5):  
            threading.Thread(target=playsound, args=(ALARM_SOUND,), daemon=True).start()  
            last_alert_time = current_time

        # ✅ Show video feed
        cv2.imshow("📹 Real-Time Phone Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("❌ Stream interrupted manually.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera stream closed.")
