import cv2
import numpy as np
import io
from flask import Flask, request, jsonify, send_file
from PIL import Image

app = Flask(__name__)

def preprocess_image(image):
    """
    Enhanced preprocessing pipeline for better edge detection
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Apply bilateral filter to preserve edges while smoothing
    bilateral = cv2.bilateralFilter(denoised, d=9, sigmaColor=75, sigmaSpace=75)
    
    return bilateral

def detect_edges(preprocessed_img):
    """
    Improved edge detection with automatic threshold calculation
    """
    # Calculate optimal thresholds using Otsu's method
    high_thresh, _ = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    
    # Apply Canny edge detection
    edges = cv2.Canny(preprocessed_img, low_thresh, high_thresh)
    
    # Enhance edges with more aggressive dilation to connect rounded corners
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    return edges

def analyze_contour_shape(contour):
    """
    Analyze contour shape to detect both rectangular and rounded rectangular shapes
    """
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.int_)  # Changed from np.int0 to np.int_
    rect_area = cv2.contourArea(box)
    
    # Get actual contour area
    actual_area = cv2.contourArea(contour)
    
    # Calculate perimeter
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate circularity (4π × area / perimeter²)
    circularity = 4 * np.pi * actual_area / (perimeter * perimeter)
    
    # Calculate solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = actual_area / hull_area if hull_area > 0 else 0
    
    # Calculate area ratio between contour and its minimum area rectangle
    area_ratio = actual_area / rect_area if rect_area > 0 else 0
    
    return {
        'circularity': circularity,
        'solidity': solidity,
        'area_ratio': area_ratio,
        'rect': rect,
        'box': box
    }

def find_card_contour(edges, image_shape):
    """
    Enhanced contour detection supporting both rectangular and rounded cards
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = image_shape[:2]
    image_area = height * width
    best_card = None
    best_score = 0
    best_box = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Basic area check
        if 0.1 * image_area <= area <= 0.7 * image_area:
            # Analyze shape characteristics
            shape_info = analyze_contour_shape(contour)
            
            # Get minimum area rectangle dimensions
            rect = shape_info['rect']
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            
            # Calculate aspect ratio score
            aspect_ratio = max(w, h) / min(w, h)
            aspect_score = 1 - min(abs(aspect_ratio - 1.586), 1.0)
            
            # Evaluate shape characteristics
            circularity_score = 1 - min(abs(shape_info['circularity'] - 0.95), 1.0)  # Perfect rectangle ≈ 0.95
            solidity_score = shape_info['solidity']  # Should be high for both rectangular and rounded
            area_ratio_score = shape_info['area_ratio']  # Should be close to 1 for rectangular, slightly less for rounded
            
            # Combined score with weights
            total_score = (
                0.3 * aspect_score +
                0.2 * circularity_score +
                0.25 * solidity_score +
                0.25 * area_ratio_score
            )
            
            if total_score > best_score:
                best_score = total_score
                best_card = contour
                best_box = shape_info['box']
    
    return (best_card, best_box) if best_score > 0.6 else (None, None)

def order_points(pts):
    """
    Improved point ordering for perspective transform
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Find top-left and bottom-right points
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Find top-right and bottom-left points
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def numpy_to_bytes(image_array, format='PNG'):
    """
    Convert numpy array to bytes for HTTP response
    """
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Save to bytes buffer
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format=format)
    img_buffer.seek(0)
    
    return img_buffer

@app.route('/crop', methods=['POST'])
def crop():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image directly from memory
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to load image'}), 400
        
        # Process image
        preprocessed = preprocess_image(image)
        edges = detect_edges(preprocessed)
        card_contour, box = find_card_contour(edges, image.shape)
        
        if card_contour is None:
            return jsonify({'error': 'No valid business card detected'}), 400
        
        # Use the minimum area rectangle points for perspective transform
        rect = order_points(box.astype(np.float32))
        
        # Calculate dimensions
        width_a = np.linalg.norm(rect[2] - rect[3])
        width_b = np.linalg.norm(rect[1] - rect[0])
        height_a = np.linalg.norm(rect[1] - rect[2])
        height_b = np.linalg.norm(rect[0] - rect[3])
        
        max_width = max(int(width_a), int(width_b))
        max_height = max(int(height_a), int(height_b))
        
        # Adjust to standard business card ratio if close
        aspect_ratio = max_width / max_height
        if 1.4 <= aspect_ratio <= 1.8:  # Close to standard ratio
            max_height = int(max_width / 1.586)  # ISO standard ratio
        
        dst_points = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
        
        # Apply perspective transform
        matrix = cv2.getPerspectiveTransform(rect, dst_points)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        
        # Convert result to bytes and return
        img_buffer = numpy_to_bytes(warped)
        
        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name='cropped_business_card.png'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)