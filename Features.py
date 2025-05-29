import cv2
import numpy as np
import easyocr

def Mean(img: np.ndarray) -> float:
    """
    Compute the mean intensity of an image.
    
    Args:
        img (np.ndarray): Input image.
        
    Returns:
        float: Mean pixel intensity.
    """
    return np.mean(img)

def Std_deviation(img: np.ndarray) -> float:
    """
    Compute the standard deviation of pixel intensities (contrast).
    
    Args:
        img (np.ndarray): Input image.
        
    Returns:
        float: Standard deviation of pixel intensities.
    """
    return np.std(img)

def extract_text(image):
    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(image)

    # Defensive check: ensure results are in expected format
    if not results or not all(len(item) == 3 for item in results):
        return "Text extraction failed or returned unexpected format."

    # Sort by Y (vertical), then X (horizontal)
    results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))

    lines = []
    current_line = []
    last_y = None

    for (bbox, text, confidence) in results:
        y = bbox[0][1]  # Top-left Y coordinate

        if last_y is None:
            last_y = y

        if abs(y - last_y) > 15:  # New line if vertical distance > threshold
            lines.append(" ".join(current_line))
            current_line = [text]
            last_y = y
        else:
            current_line.append(text)

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)
