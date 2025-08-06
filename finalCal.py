from ultralytics import YOLO
import numpy as np

# Load model
model = YOLO(r"C:\Users\Shreya\Desktop\EvolveTag\phone_detection\runs\custom_train\weights\best.pt")

# Validation
results = model.val()

# Metrics from results.box
precision = results.box.mp * 100  # Precision in %
recall = results.box.mr * 100     # Recall in %
map50 = results.box.map50 * 100   # mAP@0.5 in %
map = results.box.map * 100       # mAP@0.5:0.95 in %

# Calculate F1-Score
if (precision + recall) > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0

# FPS - average inference speed from results.speed
# results.speed is a dict with keys like 'inference' time in milliseconds
fps = None
if hasattr(results, 'speed') and 'inference' in results.speed:
    inference_time_ms = results.speed['inference']
    fps = 1000 / inference_time_ms if inference_time_ms > 0 else None

# Confusion matrix to calculate Specificity and FPR
conf_matrix = None

if hasattr(results, 'confusion') and results.confusion is not None:
    conf_matrix = results.confusion
elif hasattr(results.box, 'confusion') and results.box.confusion is not None:
    conf_matrix = results.box.confusion

specificity = None
fpr = None

if conf_matrix is not None:
    conf_matrix = np.array(conf_matrix)
    if conf_matrix.shape == (2, 2):
        tn, fp = conf_matrix[0]
        fn, tp = conf_matrix[1]

        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0

# Print results neatly
print("Metrics\tValues")
print(f"Precision\t{precision:.2f} %")
print(f"Recall\t{recall:.2f} %")
if fps is not None:
    print(f"FPS\t{fps:.2f}")
else:
    print("FPS\tNot available")
print(f"F1-Score (%)\t{f1_score:.2f} %")

if specificity is not None:
    print(f"Specificity (%)\t{specificity:.2f} %")
else:
    print("Specificity (%)\tNot available")

if fpr is not None:
    print(f"FPR (%)\t{fpr:.2f} %")
else:
    print("FPR (%)\tNot available")

print(f"mAP (%)\t{map50:.2f} %")
