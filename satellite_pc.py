import cv2
import numpy as np
import os
from flask import Flask, jsonify, Response
import time

app = Flask(__name__)

def preprocess_image(gray_img):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray_img)

# --- Template Preparation using SIFT ---

template_path = 'mars_rover_template.jpg'
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
if template is None:
    raise ValueError(f"Template image '{template_path}' not found.")

# Preprocess the template to boost contrast.
template = preprocess_image(template)
template_h, template_w = template.shape

# Initialize SIFT detector.
sift = cv2.SIFT_create()

# Compute keypoints and descriptors for the template.
kp_template, des_template = sift.detectAndCompute(template, None)
print("Template keypoints:", len(kp_template))

# Optionally, save an image with keypoints drawn for debugging.
template_kp_img = cv2.drawKeypoints(template, kp_template, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("template_keypoints.jpg", template_kp_img)

# Create a BFMatcher object using L2 norm (suitable for SIFT).
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# --- Image Capture Function ---

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

# --- Template Matching using SIFT ---

def find_template_sift(image, min_matches=5, ratio_thresh=0.75):
    """
    Detects the template in the given image using SIFT feature matching.
    
    Args:
        image: BGR image captured from the camera.
        min_matches: Minimum number of good matches required.
        ratio_thresh: Lowe's ratio threshold for filtering matches.
    
    Returns:
        center (tuple): (center_x, center_y) of the detected template region.
        corners (np.array): The four transformed corner points of the template in the scene.
        good_matches (list): The list of good matches.
        If detection fails, returns (None, None, good_matches).
    """
    # Convert the scene image to grayscale and preprocess it.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = preprocess_image(gray)
    
    # Detect keypoints and compute descriptors in the scene.
    kp_scene, des_scene = sift.detectAndCompute(gray, None)
    if des_scene is None:
        print("No descriptors found in scene.")
        return None, None, None
    
    print("Scene keypoints:", len(kp_scene))
    
    # Perform KNN matching (k=2) and apply the ratio test.
    knn_matches = bf.knnMatch(des_template, des_scene, k=2)
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    
    print("Good matches:", len(good_matches))
    
    if len(good_matches) < min_matches:
        print("Not enough good matches.")
        return None, None, good_matches
    
    # Extract the coordinates of matching keypoints.
    pts_template = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography using RANSAC.
    H, mask = cv2.findHomography(pts_template, pts_scene, cv2.RANSAC, 5.0)
    if H is None:
        print("Homography computation failed.")
        return None, None, good_matches
    
    # Define the template's corner points.
    template_corners = np.float32([[0, 0],
                                   [template_w, 0],
                                   [template_w, template_h],
                                   [0, template_h]]).reshape(-1, 1, 2)
    # Transform the template corners to the scene.
    scene_corners = cv2.perspectiveTransform(template_corners, H)
    
    # Compute the centroid of the detected region.
    center = np.mean(scene_corners, axis=0)[0]
    center = (int(center[0]), int(center[1]))
    
    return center, scene_corners, good_matches

# --- Flask Endpoints ---

@app.route('/find_rover', methods=['GET'])
def find_rover_endpoint():
    """
    Capture an image (or use the existing capture.jpg), use SIFT feature matching to detect the template,
    and return its centroid and number of good matches as JSON.
    """
    try:
        image = capture_image()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    center, corners, good_matches = find_template_sift(image)
    if center is None:
        return jsonify({"error": "Template not found"}), 404
    else:
        return jsonify({"x": center[0], "y": center[1], "good_matches": len(good_matches)})

@app.route('/render', methods=['GET'])
def render_endpoint():
    """
    Capture an image (or use the existing capture.jpg), use SIFT feature matching to detect the template,
    draw the SIFT keypoints on the scene, overlay the detected template region and centroid,
    and return the annotated image as a JPEG.
    """
    try:
        image = capture_image()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Convert the image to grayscale and preprocess.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = preprocess_image(gray)
    
    # Detect SIFT keypoints/descriptors in the scene.
    kp_scene, des_scene = sift.detectAndCompute(gray, None)
    keypoints_img = cv2.drawKeypoints(image, kp_scene, None, color=(0, 0, 255), flags=0)
    
    # Run template matching.
    center, corners, good_matches = find_template_sift(image)
    
    # Annotate with the total keypoints count.
    cv2.putText(keypoints_img, f"Keypoints: {len(kp_scene)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    if center is not None and corners is not None:
        corners_int = np.int32(corners)
        # Draw the detected template region.
        cv2.polylines(keypoints_img, [corners_int], isClosed=True, color=(0, 255, 0), thickness=2)
        # Mark the centroid.
        cv2.circle(keypoints_img, center, 5, (0, 255, 0), -1)
        # Annotate with text.
        cv2.putText(keypoints_img, f"Template ({center[0]}, {center[1]})", (center[0]-40, center[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Annotate with number of good matches.
        cv2.putText(keypoints_img, f"Good Matches: {len(good_matches)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(keypoints_img, "Template not found", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Optionally, save the annotated image.
    cv2.imwrite("annotated_capture.jpg", keypoints_img)
    
    ret, jpeg = cv2.imencode('.jpg', keypoints_img)
    if not ret:
        return jsonify({"error": "Could not encode image"}), 500
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
