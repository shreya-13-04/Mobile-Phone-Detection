from ultralytics import YOLO

# Load base YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model on your dataset
model.train(
    data="data.yaml",         # Path to your YAML file
    epochs=50,                # Number of training epochs
    imgsz=640,                # Image size
    project="runs",           # Project directory
    name="custom_train",      # Name of the run
    batch=16,                 # Adjust depending on your memory
    device="cpu"              # Or "0" if using GPU
)
