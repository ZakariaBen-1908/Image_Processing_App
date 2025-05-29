import cv2
import streamlit as st

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
