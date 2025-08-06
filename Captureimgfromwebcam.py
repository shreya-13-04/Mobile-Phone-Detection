import cv2
import os

save_dir = "images/test"  #Folder where test images will go
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while count < 10:  # Capture 10 test images (you can change)
    ret, frame = cap.read()
    if not ret:
        break

    filename = os.path.join(save_dir, f"test_{count}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Saved {filename}")
    count += 1

    cv2.imshow("Capturing Test Images", frame)
    if cv2.waitKey(500) & 0xFF == ord('q'):  # Capture every 500ms
        break

cap.release()
cv2.destroyAllWindows()
