import cv2
import numpy as np
from flask import Flask, jsonify, Response
import time

app = Flask(__name__)

# Load the template image (grayscale) and precompute its ORB keypoints and descriptors.
template = cv2.imread('mars_rover_body_template.jpg', cv2.IMREAD_GRAYSCALE)
if template is None:
    raise ValueError("Template image 'mars_rover_template' not found.")
template_h, template_w = template.shape

# Initialize ORB detector.
orb = cv2.ORB_create()

# Compute keypoints and descriptors for the template.
kp_template, des_template = orb.detectAndCompute(template, None)

# Create a BFMatcher object with Hamming distance (suitable for ORB).
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def capture_image():
    """
    Captures a single image from an alternative camera (device index 1)
    and returns it as a BGR NumPy array.
    """
    print("capturing...")
    cap = cv2.VideoCapture(1)  # Adjust index if needed.
    if not cap.isOpened():
        raise RuntimeError("Could not open camera on device index 1")
    
    # Allow the camera to warm up.
    time.sleep(0.5)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError("Failed to capture image from the camera")
    print("capturing done.")
    return frame

def find_template_orb(image, min_matches=10):
    """
    Uses ORB feature matching to locate the template in the given image.
    
    Args:
        image: BGR image captured from the camera.
        min_matches: Minimum number of matches required to consider the detection valid.
        
    Returns:
        center (tuple): (center_x, center_y) of the detected template region.
        corners (np.array): The four corners of the detected template region in the scene.
        Or (None, None) if not enough matches are found or homography fails.
    """
    # Convert captured image to grayscale.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and compute descriptors in the scene.
    kp_scene, des_scene = orb.detectAndCompute(gray_image, None)
    if des_scene is None:
        return None, None

    # Match descriptors between the template and scene.
    matches = bf.match(des_template, des_scene)
    
    # Sort matches by distance. (Lower distance is better.)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < min_matches:
        return None, None
    
    # Extract location of good matches.
    pts_template = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute homography if possible.
    H, mask = cv2.findHomography(pts_template, pts_scene, cv2.RANSAC, 5.0)
    if H is None:
        return None, None

    # Define the corners of the template.
    template_corners = np.float32([[0, 0],
                                   [template_w, 0],
                                   [template_w, template_h],
                                   [0, template_h]]).reshape(-1, 1, 2)
    
    # Transform the template corners to the scene using the homography.
    scene_corners = cv2.perspectiveTransform(template_corners, H)
    
    # Compute the centroid (center) of the detected region.
    center = np.mean(scene_corners, axis=0)[0]
    center = (int(center[0]), int(center[1]))
    
    return center, scene_corners

@app.route('/find_rover', methods=['GET'])
def find_rover_endpoint():
    """
    Capture an image from the camera, use ORB feature matching to detect the template,
    and return the center coordinates as JSON.
    """
    try:
        image = capture_image()
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    center, corners = find_template_orb(image)
    if center is None:
        return jsonify({'error': 'Template not found'}), 404
    else:
        return jsonify({'x': center[0], 'y': center[1]})

@app.route('/render', methods=['GET'])
def render_endpoint():
    """
    Capture an image from the camera, use ORB feature matching to detect the template,
    annotate the image with the detected region and its center, and return the annotated image.
    """
    try:
        image = capture_image()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    center, corners = find_template_orb(image)
    
    if center is not None and corners is not None:
        # Draw the detected region (polygon) on the image.
        corners_int = np.int32(corners)
        cv2.polylines(image, [corners_int], isClosed=True, color=(0, 255, 0), thickness=2)
        # Draw a circle at the centroid.
        cv2.circle(image, center, 5, (0, 255, 0), -1)
        # Annotate the image with text.
        cv2.putText(image, f'Template ({center[0]}, {center[1]})', (center[0]-40, center[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(image, 'Template not found', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Encode the annotated image as JPEG.
    ret, jpeg = cv2.imencode('.jpg', image)
    if not ret:
        return jsonify({'error': 'Could not encode image'}), 500
    
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
