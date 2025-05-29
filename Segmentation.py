import cv2
import numpy as np
import streamlit as st

def Thresholding(img):
    t = st.slider("Select Threshold Value:", 0, 255, 100)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(img_gray, t, 255, cv2.THRESH_BINARY)
    return th

def Adaptive_Thresholding(img):
    """Apply adaptive thresholding."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    block_size = st.slider("Block Size (odd number)", 3, 51, 11, step=2)
    c_value = st.slider("C (constant subtracted)", 0, 20, 2)
    adaptive = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, block_size, c_value)
    return adaptive

def Otsu_Segmentation(img):
    """Automatically segment image using Otsu's method."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return segmented

def Sobel(img):
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    i=cv2.filter2D(img, -1, kernel)
    return i

def Canny(img):
    threshold1 = st.slider("Canny Threshold 1:", 50, 300, 100)
    threshold2 = st.slider("Canny Threshold 2:", 50, 300, 200)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a=cv2.Canny(img_gray, threshold1, threshold2)
    return a

def Watershed_Segmentation(img):
    """Use watershed algorithm for precise segmentation of overlapping objects."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    img_color = img.copy()
    markers = cv2.watershed(img_color, markers)
    img_color[markers == -1] = [255, 0, 0]  # Mark boundaries in red

    return img_color

def Kmeans_Segmentation(img):
    """Segment image using K-means clustering."""
    K = st.slider("Number of segments (K):", 2, 10, 4)
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_img = centers[labels.flatten()].reshape((img.shape))
    return segmented_img

