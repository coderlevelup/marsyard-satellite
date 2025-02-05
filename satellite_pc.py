import cv2
import numpy as np
from flask import Flask, jsonify, Response
import time

app = Flask(__name__)

def capture_image():
    """
    Captures a single image from the webcam (device index 1)
    and returns it as a BGR NumPy array.
    """
    cap = cv2.VideoCapture(1)  # Adjust index if needed.
    if not cap.isOpened():
        raise RuntimeError("Could not open camera on device index 1")
    time.sleep(0.5)  # Allow the camera to warm up.
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture image from the camera")
    return frame

def detect_light_gray_cluster(image, min_area=50, max_area=500):
    """
    Detects a small cluster of light gray pixels in the image using basic thresholding.

    Process:
      1. Convert the image to HSV.
      2. Threshold for light gray (low saturation and high brightness).
      3. Optionally apply morphological operations to reduce noise.
      4. Find contours in the binary mask.
      5. For each contour, if its area is between min_area and max_area, compute its centroid.
    
    Returns:
      (cx, cy, contour): The centroid (cx, cy) of the detected cluster and the contour.
      If no valid cluster is found, returns (None, None, None).
    """
    # Convert image to HSV color space.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define HSV thresholds for light gray.
    # Light gray typically has very low saturation and high brightness.
    lower_gray = np.array([0, 0, 200])    # You may adjust these values as needed.
    upper_gray = np.array([179, 50, 255])
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Optionally reduce noise with morphological operations.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours in the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for a contour with area within our desired range.
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            # Compute centroid of the contour.
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return cx, cy, cnt
    return None, None, None

@app.route('/find_rover', methods=['GET'])
def find_rover_endpoint():
    """
    Capture an image, detect a small light gray cluster, and return its centroid as JSON.
    """
    try:
        image = capture_image()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    cx, cy, _ = detect_light_gray_cluster(image)
    if cx is None:
        return jsonify({"error": "Light gray cluster not found"}), 404
    else:
        return jsonify({"x": cx, "y": cy})

@app.route('/render', methods=['GET'])
def render_endpoint():
    """
    Capture an image, detect a small light gray cluster, annotate the image with the
    detected contour and centroid, and return the annotated image as a JPEG.
    """
    try:
        image = capture_image()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    cx, cy, contour = detect_light_gray_cluster(image)
    if cx is not None and contour is not None:
        # Draw the detected contour.
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        # Draw a circle at the centroid.
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
        # Annotate with text.
        cv2.putText(image, f"Cluster ({cx}, {cy})", (cx - 40, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(image, "Light gray cluster not found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    ret, jpeg = cv2.imencode(".jpg", image)
    if not ret:
        return jsonify({"error": "Could not encode image"}), 500
    return Response(jpeg.tobytes(), mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
