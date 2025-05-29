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

def extract_text(img: np.ndarray, lang_list=['en'], use_gpu=False) -> list:
    """
    Extract text from an image using EasyOCR.
    
    Args:
        img (np.ndarray): Input image.
        lang_list (list): List of language codes (default is ['en']).
        use_gpu (bool): Whether to use GPU (default is False).
    
    Returns:
        list: List of tuples (bounding_box, text, confidence).
    """
    reader = easyocr.Reader(lang_list, gpu=use_gpu)
    results = reader.readtext(img)
    return results
