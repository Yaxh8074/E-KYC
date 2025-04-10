import cv2
import os
import numpy as np
import logging
from io import BytesIO

# Set up logging
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "ekyc_logs.log"), 
    level=logging.INFO, 
    format=logging_str, 
    filemode="a"
)

def select_best_frame(frames):
    """
    Select the best frame from a sequence of video frames
    Uses face detection and blur detection to pick the clearest face
    
    Parameters:
    - frames: List of video frames
    
    Returns:
    - Best frame or None if no suitable frame found
    """
    if not frames:
        return None
        
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    best_score = -1
    best_frame = None
    
    for frame in frames:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            continue
            
        # Find largest face
        largest_face = None
        largest_area = 0
        
        for (x, y, w, h) in faces:
            if w * h > largest_area:
                largest_area = w * h
                largest_face = (x, y, w, h)
                
        if largest_face is None:
            continue
            
        # Extract face region
        (x, y, w, h) = largest_face
        face_region = gray[y:y+h, x:x+w]
        
        # Calculate blur score (higher variance = less blurry)
        laplacian_var = cv2.Laplacian(face_region, cv2.CV_64F).var()
        
        # Calculate overall score (combination of face size and clarity)
        score = largest_area * laplacian_var
        
        if score > best_score:
            best_score = score
            best_frame = frame
            
    # If no good frame found, return middle frame as fallback
    if best_frame is None and frames:
        return frames[len(frames) // 2]
        
    return best_frame

def enhance_selfie(image):
    """
    Enhance a selfie image with basic improvements
    
    Parameters:
    - image: Input image
    
    Returns:
    - Enhanced image
    """
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge((l, a, b))
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Subtle sharpening
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    except Exception as e:
        logging.error(f"Error enhancing selfie: {e}")
        return image  # Return original if enhancement fails

def frame_to_bytes(frame):
    """
    Convert OpenCV frame to bytes for Streamlit display
    
    Parameters:
    - frame: OpenCV image frame
    
    Returns:
    - BytesIO object containing the image data
    """
    try:
        # Encode frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        
        # Convert to BytesIO object
        return BytesIO(buffer)
    except Exception as e:
        logging.error(f"Error converting frame to bytes: {e}")
        return None