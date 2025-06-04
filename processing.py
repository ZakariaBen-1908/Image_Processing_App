import cv2
import streamlit as st
import numpy as np

def Con_to_grey(img):
    """Convert image to grey."""
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

def Histogram_Equalization(img):
    """Apply histogram equalization (grey only)."""
    grey = Con_to_grey(img)
    equalized = cv2.equalizeHist(grey)
    return equalized

def NLM_Denoising(img, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    """Apply Non-Local Means Denoising and return denoised, noise, and difference images."""
    denoised = cv2.fastNlMeansDenoisingColored(
        img, None, h, hColor, templateWindowSize, searchWindowSize
    )
    # Convert to grayscale for diff computation (optional)
    gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    # Calculate difference
    diff = cv2.absdiff(img, denoised)
    noise = cv2.absdiff(gray_original, gray_denoised)

    return denoised, noise, diff


def Sharpening(img):
    """Apply sharpening using kernel filter."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
