import cv2
import numpy as np
import os
from flask import Flask, jsonify, Response
import time
from ultralytics import YOLO

app = Flask(__name__)

# --- YOLOv8 Custom Model Setup ---
# Load your custom YOLOv8 model from 'best.pt'
model = YOLO("best.pt")

# Set your target class name as defined in your custom model.
# For example, if your model's class names include "rover", set it accordingly.
TARGET_CLASS = "rover"

def capture_image():
    """
    Checks if 'capture.jpg' exists; if so, loads and returns that image.
    Otherwise, captures an image from the webcam (device index 1), saves it as 'capture.jpg',
    and returns the captured image.
    """
    filename = "capture.jpg"
    if os.path.exists(filename):
        print(f"Using existing file: {filename}")
        image = cv2.imread(filename)
        if image is None:
            raise RuntimeError("Failed to load image from file.")
        return image
    else:
        print("Capture file not found. Capturing new image from webcam.")
        cap = cv2.VideoCapture(1)  # Adjust index if needed.
        if not cap.isOpened():
            raise RuntimeError("Could not open camera on device index 1")
        time.sleep(0.5)  # Allow the camera to warm up.
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to capture image from the camera")
        cv2.imwrite(filename, frame)
        return frame

def detect_target_yolo(image, conf_threshold=0.5, target_class=TARGET_CLASS):
    """
    Uses the YOLOv8 custom model to detect objects in the image and returns the bounding box and
    centroid of the detection that matches the target class with the highest confidence.
    
    Args:
        image: BGR image from the camera.
        conf_threshold: Minimum confidence threshold for detections.
        target_class: The class name of the target object.
    
    Returns:
        center (tuple): (center_x, center_y) of the detected target.
        box (tuple): (x, y, w, h) bounding box of the detected target.
        If no target is found, returns (None, None).
    """
    # Run inference using the YOLOv8 model.
    results = model(image, conf=conf_threshold)[0]  # Process the first (and only) image result

    target_box = None
    target_conf = -1
    target_center = None

    # Iterate through detections.
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()  # Each box: [x1, y1, x2, y2]
        confidences = results.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Class indices

        # Loop through detections and select the target class with the highest confidence.
        for i in range(len(boxes)):
            class_name = model.names[class_ids[i]]
            if class_name == target_class:
                if confidences[i] > target_conf:
                    target_conf = confidences[i]
                    x1, y1, x2, y2 = boxes[i]
                    target_box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    target_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    return target_center, target_box

@app.route('/find_rover', methods=['GET'])
def find_rover_endpoint():
    """
    Capture an image (or use 'capture.jpg' if it exists), use the YOLOv8 custom model to detect
    the target object, and return its centroid coordinates as JSON.
    """
    try:
        image = capture_image()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    center, box = detect_target_yolo(image)
    if center is None:
        return jsonify({"error": "Target object not found"}), 404
    else:
        return jsonify({"x": center[0], "y": center[1]})

@app.route('/render', methods=['GET'])
def render_endpoint():
    """
    Capture an image (or use 'capture.jpg' if it exists), use the YOLOv8 custom model to detect
    the target object, annotate the image with the bounding box and centroid, and return the annotated image as JPEG.
    """
    try:
        image = capture_image()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    center, box = detect_target_yolo(image)
    
    if center is not None and box is not None:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(image, center, 5, (0, 255, 0), -1)
        cv2.putText(image, f"{TARGET_CLASS} ({center[0]}, {center[1]})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(image, "Target object not found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    ret, jpeg = cv2.imencode(".jpg", image)
    if not ret:
        return jsonify({"error": "Could not encode image"}), 500
    return Response(jpeg.tobytes(), mimetype="image/jpeg")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
