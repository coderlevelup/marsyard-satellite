import cv2
import numpy as np
from flask import Flask, jsonify, Response
import time

app = Flask(__name__)

def preprocess_image(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

# Load and preprocess the template image.
template = cv2.imread('mars_rover_template.jpg', cv2.IMREAD_GRAYSCALE)
if template is None:
    raise ValueError("Template image 'mars_rover_template.jpg' not found.")
template = preprocess_image(template)
template_h, template_w = template.shape

# Initialize SIFT detector.
sift = cv2.SIFT_create()

# Compute keypoints and descriptors for the template.
kp_template, des_template = sift.detectAndCompute(template, None)

# Create a BFMatcher object using L2 norm.
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

def capture_image():
    cap = cv2.VideoCapture(1)  # Adjust index if needed.
    if not cap.isOpened():
        raise RuntimeError("Could not open camera on device index 1")
    time.sleep(0.5)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture image from the camera")
    return frame

def find_template_sift(image, min_matches=10, ratio_thresh=0.85):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = preprocess_image(gray_image)
    kp_scene, des_scene = sift.detectAndCompute(gray_image, None)
    if des_scene is None:
        return None, None
    knn_matches = bf.knnMatch(des_template, des_scene, k=2)
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    print(f"Found {len(good_matches)} good matches.")  # Debug print
    if len(good_matches) < min_matches:
        return None, None
    pts_template = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts_template, pts_scene, cv2.RANSAC, 5.0)
    if H is None:
        return None, None
    template_corners = np.float32([[0, 0],
                                   [template_w, 0],
                                   [template_w, template_h],
                                   [0, template_h]]).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(template_corners, H)
    center = np.mean(scene_corners, axis=0)[0]
    center = (int(center[0]), int(center[1]))
    return center, scene_corners

@app.route('/find_rover', methods=['GET'])
def find_rover_endpoint():
    try:
        image = capture_image()
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    center, corners = find_template_sift(image)
    if center is None:
        return jsonify({'error': 'Template not found'}), 404
    else:
        return jsonify({'x': center[0], 'y': center[1]})

@app.route('/render', methods=['GET'])
def render_endpoint():
    try:
        image = capture_image()
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    center, corners = find_template_sift(image)
    if center is not None and corners is not None:
        corners_int = np.int32(corners)
        cv2.polylines(image, [corners_int], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.circle(image, center, 5, (0, 255, 0), -1)
        cv2.putText(image, f'Template ({center[0]}, {center[1]})', (center[0]-40, center[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(image, 'Template not found', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    ret, jpeg = cv2.imencode('.jpg', image)
    if not ret:
        return jsonify({'error': 'Could not encode image'}), 500
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
