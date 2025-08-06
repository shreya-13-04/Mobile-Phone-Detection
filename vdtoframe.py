import cv2
import os

# Define the video file path and output folder
video_path = "wop.mp4"  # Replace with your video file path
output_folder = "frames2"  # Replace with your desired output folder

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Frame counter
frame_number = 0

# Read frames from the video
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if no frames are left
    
    # Save the frame as an image
    frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    print(f"Saved: {frame_filename}")
    
    frame_number += 1

# Release the video capture object
cap.release()

print(f"All frames saved in folder: {output_folder}")
