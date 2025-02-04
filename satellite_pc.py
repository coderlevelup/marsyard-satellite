import cv2
import numpy as np
from flask import Flask, jsonify, Response
import time

app = Flask(__name__)

# --- YOLO MODEL SETUP ---
YOLO_CONFIG = "yolov3.cfg"
YOLO_WEIGHTS = "yolov3.weights"
YOLO_NAMES = "coco.names"

# Load class names from the coco.names file.
with open(YOLO_NAMES, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set your target class. Change this to a class present in coco.names.
TARGET_CLASS = "truck"

# Load YOLO model.
net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG, YOLO_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the output layer names.
layer_names = net.getLayerNames()
# Some versions of OpenCV return a tuple for getUnconnectedOutLayers().
unconnected_layers = net.getUnconnectedOutLayers()
if isinstance(unconnected_layers, tuple):
    output_layers = [layer_names[i - 1] for i in unconnected_layers]
else:
    output_layers = [layer_names[i - 1] for i in unconnected_layers.flatten()]

# --- IMAGE CAPTURE FUNCTION ---
def capture_image():
    """
    Captures a single image from an alternative camera (device index 1)
    and returns it as a BGR NumPy array.
    """
    cap = cv2.VideoCapture(1)  # Change index if needed.
    if not cap.isOpened():
        raise RuntimeError("Could not open camera on device index 1")
    
    time.sleep(0.5)  # Allow the camera to warm up.
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError("Failed to capture image from the camera")
    
    return frame

# --- YOLO DETECTION FUNCTION ---
def detect_target_yolo(image, conf_threshold=0.5, nms_threshold=0.4):
    """
    Uses YOLO to detect objects in the image and returns the bounding box and centroid
    of the detection matching TARGET_CLASS with the highest confidence.
    
    Args:
        image: BGR image from the camera.
        conf_threshold: Confidence threshold for detections.
        nms_threshold: Non-maxima suppression threshold.
    
    Returns:
        center (tuple): (center_x, center_y) of the detected object.
        box (tuple): (x, y, w, h) bounding box of the detected object.
        If no target is found, returns (None, None).
    """
    height, width = image.shape[:2]
    # Create a blob from the image.
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    
    # Loop over each output layer.
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maxima suppression.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:
        indices = np.array(indices).flatten()
    else:
        indices = []
    
    best_box = None
    best_conf = -1
    # Look for detections matching our target class.
    for i in indices:
        if classes[class_ids[i]] == TARGET_CLASS:
            if confidences[i] > best_conf:
                best_conf = confidences[i]
                best_box = boxes[i]
    
    if best_box is not None:
        x, y, w, h = best_box
        center = (x + w // 2, y + h // 2)
        return center, best_box
    else:
        return None, None

# --- FLASK ENDPOINTS ---
@app.route('/find_rover', methods=['GET'])
def find_rover_endpoint():
    """
    Capture an image from the camera, use YOLO to detect the target object,
    and return its center coordinates as JSON.
    """
    try:
        image = capture_image()
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    center, box = detect_target_yolo(image)
    if center is None:
        return jsonify({'error': 'Target object not found'}), 404
    else:
        return jsonify({'x': center[0], 'y': center[1]})

@app.route('/render', methods=['GET'])
def render_endpoint():
    """
    Capture an image from the camera, use YOLO to detect the target object,
    annotate the image with a bounding box and the center coordinate,
    and return the annotated image as a JPEG.
    """
    try:
        image = capture_image()
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    center, box = detect_target_yolo(image)
    
    if center is not None and box is not None:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(image, center, 5, (0, 255, 0), -1)
        cv2.putText(image, f'{TARGET_CLASS} ({center[0]}, {center[1]})', 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(image, 'Target object not found', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    ret, jpeg = cv2.imencode('.jpg', image)
    if not ret:
        return jsonify({'error': 'Could not encode image'}), 500
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
