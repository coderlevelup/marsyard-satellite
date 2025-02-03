import cv2
import numpy as np
from flask import Flask, jsonify, Response
import time

app = Flask(__name__)

# Load the template image once when the application starts.
# The template should be a cropped grayscale image of your object.
template = cv2.imread('mars_rover_template.jpg', cv2.IMREAD_GRAYSCALE)
if template is None:
    raise ValueError("Template image 'mars_rover_template.jpg' not found.")
template_w, template_h = template.shape[::-1]

def capture_image():
    """
    Captures a single image from an alternative camera (device index 1)
    and returns it as a BGR NumPy array.
    """
    cap = cv2.VideoCapture(1)  # Adjust the index if needed
    if not cap.isOpened():
        raise RuntimeError("Could not open camera on device index 1")
    
    # Allow the camera to warm up
    time.sleep(0.5)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError("Failed to capture image from the camera")
    
    return frame

def find_template_in_image(image, threshold=0.7):
    """
    Uses template matching to detect the template in the given image.
    
    Args:
        image: The BGR image captured from the camera.
        threshold: The minimum matching score (0 to 1) to consider a valid detection.
    
    Returns:
        (center_x, center_y): The center coordinates of the best-matching region,
        or (None, None) if the match score is below the threshold.
    """
    # Convert the captured image to grayscale for matching.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply template matching.
    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val >= threshold:
        # Calculate the center point of the detected template.
        top_left = max_loc
        center_x = top_left[0] + template_w // 2
        center_y = top_left[1] + template_h // 2
        return center_x, center_y, top_left
    else:
        return None, None, None

@app.route('/find_rover', methods=['GET'])
def find_rover_endpoint():
    """
    Capture an image from the camera, apply template matching to detect the object,
    and return the center coordinates as JSON.
    """
    try:
        image = capture_image()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    x, y, _ = find_template_in_image(image)
    if x is None:
        return jsonify({'error': 'Template not found'}), 404
    else:
        return jsonify({'x': x, 'y': y})

@app.route('/render', methods=['GET'])
def render_endpoint():
    """
    Capture an image from the camera, apply template matching to detect the object,
    annotate the image with a rectangle around the matching region and its center,
    and return the annotated image as a JPEG.
    """
    try:
        image = capture_image()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    x, y, top_left = find_template_in_image(image)
    
    if x is not None and top_left is not None:
        # Draw a rectangle around the detected template region.
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        # Mark the center with a circle.
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        # Annotate the image with text.
        cv2.putText(image, f'Template ({x}, {y})', (x - 40, y - 10),
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
