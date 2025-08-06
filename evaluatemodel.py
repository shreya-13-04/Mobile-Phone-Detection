from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Or your custom-trained model
results = model.val(data="data.yaml", split="test")
