import os
import cv2
import numpy as np
from ultralytics import YOLO

def xywhn_to_xyxy(xywhn, w, h):
    # Converts normalized xywh to xyxy format (absolute coords)
    x, y, bw, bh = xywhn
    x1 = (x - bw / 2) * w
    y1 = (y - bh / 2) * h
    x2 = (x + bw / 2) * w
    y2 = (y + bh / 2) * h
    return [x1, y1, x2, y2]

def load_ground_truths(label_path, img_w, img_h):
    boxes = []
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(label_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if cls == 0:  # assuming class 0 is phone
                xywhn = list(map(float, parts[1:5]))
                boxes.append(xywhn_to_xyxy(xywhn, img_w, img_h))
    return boxes


def iou(boxA, boxB):
    # box format: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_model(model, img_folder, label_folder, iou_threshold=0.5):
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    img_files = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for img_file in img_files:
        img_path = os.path.join(img_folder, img_file)
        label_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + '.txt')

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        gt_boxes = load_ground_truths(label_path, w, h)
        results = model(img_path)
        pred_boxes = []

        for det in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, cls = det
            if int(cls) == 0 and score > 0.25:  # threshold and class filter
                pred_boxes.append([x1, y1, x2, y2])

        matched_gt = set()
        matched_pred = set()

        # Match predictions to ground truths
        for i, gt_box in enumerate(gt_boxes):
            matched = False
            for j, pred_box in enumerate(pred_boxes):
                if j in matched_pred:
                    continue
                if iou(gt_box, pred_box) >= iou_threshold:
                    TP += 1
                    matched_gt.add(i)
                    matched_pred.add(j)
                    matched = True
                    break
            if not matched:
                FN += 1

        # Unmatched predictions are false positives
        FP += len(pred_boxes) - len(matched_pred)

        # For images with no phones (no GT boxes)
        if len(gt_boxes) == 0:
            if len(pred_boxes) == 0:
                TN += 1
            else:
                FP += len(pred_boxes)

    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # You can add mAP calculation from ultralytics results if you want separately

    print(f"Precision: {precision * 100:.2f} %")
    print(f"Recall: {recall * 100:.2f} %")
    print(f"Specificity: {specificity * 100:.2f} %")
    print(f"FPR: {fpr * 100:.2f} %")
    print(f"F1-Score: {f1_score * 100:.2f} %")
    print(f"True Positives: {TP}")
    print(f"False Positives: {FP}")
    print(f"True Negatives: {TN}")
    print(f"False Negatives: {FN}")

if __name__ == "__main__":
    model = YOLO(r"C:\Users\Shreya\Desktop\EvolveTag\phone_detection\runs\custom_train\weights\best.pt")
    img_folder = r"C:\Users\Shreya\Desktop\EvolveTag\phone_detection\images\test"
    label_folder = r"C:\Users\Shreya\Desktop\EvolveTag\phone_detection\labels\test"

    evaluate_model(model, img_folder, label_folder)
