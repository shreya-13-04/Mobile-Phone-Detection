import cv2
import numpy as np
import os
import time
from playsound import playsound  


BASE_DIR = r"C:\Users\Shreya\Desktop\EvolveTag\phone_detection"
LABELS_PATH = os.path.join(BASE_DIR, "coco.names")
CFG_PATH = os.path.join(BASE_DIR, "yolov4.cfg")
WEIGHTS_PATH = os.path.join(BASE_DIR, "yolov4.weights")
ALARM_SOUND = os.path.join(BASE_DIR, "warning.mp3")


net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open(LABELS_PATH, "r") as f:
    labels = f.read().strip().split("\n")


video_source = os.path.join(BASE_DIR, "contvideo1.mp4") #change to 0 if webcam
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Unable to open video source.")
    exit()

last_alert_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to fetch frame. Ending stream.")
            break

    
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        phone_detected = False  

        
        height, width = frame.shape[:2]
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

        
        current_time = time.time()
        if phone_detected and (current_time - last_alert_time > 5): 
            print("Phone Detected! Triggering Alarm...")
            playsound(ALARM_SOUND, block=False)
            last_alert_time = current_time

   
        cv2.imshow("Live Surveillance Feed", frame)

       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stream interrupted manually.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Video stream closed.")
