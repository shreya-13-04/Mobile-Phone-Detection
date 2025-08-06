import cv2
import os

# CONFIG
image_dir = "images/test"
label_dir = "labels/test"
os.makedirs(label_dir, exist_ok=True)

def draw_and_label(image_path, label_path):
    img = cv2.imread(image_path)
    clone = img.copy()
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                cv2.rectangle(clone, points[0], points[1], (0, 255, 0), 2)
                cv2.imshow("Label", clone)
                write_label(image_path, label_path, x1, y1, x2, y2)

    cv2.imshow("Label", clone)
    cv2.setMouseCallback("Label", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_label(img_path, label_path, x1, y1, x2, y2):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Convert to YOLO format
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width = abs(x2 - x1) / w
    height = abs(y2 - y1) / h

    with open(label_path, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    print(f"✅ Saved label: {label_path}")

# Run on all test images
for fname in os.listdir(image_dir):
    if fname.endswith(".jpg") or fname.endswith(".png"):
        img_path = os.path.join(image_dir, fname)
        label_path = os.path.join(label_dir, fname.replace(".jpg", ".txt").replace(".png", ".txt"))
        print(f"\nLabeling: {fname}")
        draw_and_label(img_path, label_path)
