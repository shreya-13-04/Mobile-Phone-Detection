import cv2

# Open the video stream (use 0 for webcam or replace with a video file path or RTSP URL)
video_source = "phone_detection/contvideo.mp4"
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Unable to open video source.")
    exit()

# Frame counter (optional)
frame_count = 0

try:
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Unable to fetch frame. Ending stream.")
            break
        
        frame_count += 1
        
        # Process the frame (e.g., display, save, or analyze)
        cv2.imshow('Live Surveillance Feed', frame)
        
        # Save the frame (optional)
        frame_filename = f"frame_{frame_count:06d}.jpg"
        cv2.imwrite(frame_filename, frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stream interrupted manually.")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Video stream closed.")
