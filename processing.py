import cv2
import streamlit as st
import numpy as np

def Con_to_grey(img):
    """Convert image to greyCon_to_grey."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grey

def Median_Filter(img, ksize=5):
    """Apply median filter to the image."""
    filtered = cv2.medianBlur(img, ksize)
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

def Gaussian_Filter(img, ksize=(5, 5), sigma=0):
    """Apply Gaussian blur to the image."""
    blurred = cv2.GaussianBlur(img, ksize, sigma)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

def Resize_Image(img):
    """Resize image using Streamlit sliders for width and height."""
    width = st.slider("Select width:", 50, 800, 300)
    height = st.slider("Select height:", 50, 800, 300)
    resized = cv2.resize(img, (width, height))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

def Canny_edge_detection(img):
    """Apply Canny edge detection."""
    threshold1 = st.slider("Canny Threshold 1", 50, 300, 100)
    threshold2 = st.slider("Canny Threshold 2", 50, 300, 200)
    grey = Con_to_grey(img)
    edges = cv2.Canny(grey, threshold1, threshold2)
    return edges

def Thresholding(img):
    """Apply binary thresholding."""
    grey = Con_to_grey(img)
    thresh_val = st.slider("Threshold Value", 0, 255, 127)
    _, thresh = cv2.threshold(grey, thresh_val, 255, cv2.THRESH_BINARY)
    return thresh

def Adaptive_Thresholding(img):
    """Apply adaptive thresholding."""
    grey = Con_to_grey(img)
    block_size = st.slider("Block Size (odd number)", 3, 51, 11, step=2)
    c_value = st.slider("C (constant subtracted)", 0, 20, 2)
    adaptive = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, block_size, c_value)
    return adaptive

def Histogram_Equalization(img):
    """Apply histogram equalization (greyCon_to_grey only)."""
    grey = Con_to_grey(img)
    equalized = cv2.equalizeHist(grey)
    return equalized

def Sharpening(img):
    """Apply sharpening using kernel filter."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
