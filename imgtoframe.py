import cv2

# Print OpenCV version
print("OpenCV version:", cv2.__version__)

# Load and display an image
image = cv2.imread('img1.jpeg')  # Replace with an actual image path
cv2.imshow('Test_Image1', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
