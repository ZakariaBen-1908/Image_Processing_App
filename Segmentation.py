import cv2
import numpy as np
import streamlit as st

def Thresholding(img):
    t = st.slider("Select Threshold Value:", 0, 255, 100)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(img_gray, t, 255, cv2.THRESH_BINARY)
    return th

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

