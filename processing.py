import cv2
import streamlit as st
import numpy as np

def convert_to_greyscale(img):
    """Convert image to greyscale."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grey

def apply_median_filter(img, ksize=5):
    """Apply median filter to the image."""
    filtered = cv2.medianBlur(img, ksize)
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

def apply_gaussian_filter(img, ksize=(5, 5), sigma=0):
    """Apply Gaussian blur to the image."""
    blurred = cv2.GaussianBlur(img, ksize, sigma)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

def resize_image(img):
    """Resize image using Streamlit sliders for width and height."""
    width = st.slider("Select width:", 50, 800, 300)
    height = st.slider("Select height:", 50, 800, 300)
    resized = cv2.resize(img, (width, height))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

def apply_canny_edge_detection(img):
    """Apply Canny edge detection."""
    threshold1 = st.slider("Canny Threshold 1", 50, 300, 100)
    threshold2 = st.slider("Canny Threshold 2", 50, 300, 200)
    grey = convert_to_greyscale(img)
    edges = cv2.Canny(grey, threshold1, threshold2)
    return edges

def apply_thresholding(img):
    """Apply binary thresholding."""
    grey = convert_to_greyscale(img)
    thresh_val = st.slider("Threshold Value", 0, 255, 127)
    _, thresh = cv2.threshold(grey, thresh_val, 255, cv2.THRESH_BINARY)
    return thresh

def apply_adaptive_thresholding(img):
    """Apply adaptive thresholding."""
    grey = convert_to_greyscale(img)
    block_size = st.slider("Block Size (odd number)", 3, 51, 11, step=2)
    c_value = st.slider("C (constant subtracted)", 0, 20, 2)
    adaptive = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, block_size, c_value)
    return adaptive

def apply_histogram_equalization(img):
    """Apply histogram equalization (greyscale only)."""
    grey = convert_to_greyscale(img)
    equalized = cv2.equalizeHist(grey)
    return equalized

def apply_sharpening(img):
    """Apply sharpening using kernel filter."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
