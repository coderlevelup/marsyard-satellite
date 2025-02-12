import cv2
import time
import os

# Set the camera index (0 for the default camera, 1 for the second camera)
camera_index = 1  
capture_duration = 100  # Duration in seconds
save_folder = "rover"

# Create a folder to save images if it doesn't exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Open the camera
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

print(f"Capturing images from camera {camera_index} for {capture_duration} seconds...")

for i in range(100):
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    filename = os.path.join(save_folder, f"image_{i+1:03d}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Saved {filename}")

    time.sleep(1)  # Wait for 1 second

# Release the camera
cap.release()
cv2.destroyAllWindows()
print("Image capture completed.")
