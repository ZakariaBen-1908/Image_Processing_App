import os
import cv2
import streamlit as st
import numpy as np
from PIL import Image
import processing
import Segmentation

IMAGE_DIR = "images"
OUTPUT_DIR = "Output_Images"

def ensure_directories():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_saved_images():
    return [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]

def save_image(img):
    ensure_directories()
    index = 1
    while os.path.exists(f"{IMAGE_DIR}/img{index}.jpg"):
        index += 1
    image_path = f"{IMAGE_DIR}/img{index}.jpg"
    cv2.imwrite(image_path, img)
    return image_path

def create_download_button(image, filename, label):
    filepath = os.path.join(OUTPUT_DIR, filename)
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(filepath)
    with open(filepath, "rb") as f:
        st.download_button(label=label, data=f, file_name=filename, mime="image/jpeg")

def load_selected_image(saved_images, key=None):
    selected_image = st.selectbox("Choose an image", saved_images, key=key)
    if selected_image:
        image_path = os.path.join(IMAGE_DIR, selected_image)
        img = cv2.imread(image_path)
        return img
    return None

def show_image(img, caption=None):
    img_disp = cv2.resize(img, (340, 180))
    st.image(img_disp, channels='BGR', caption=caption, use_column_width=True)

def image_acquisition():
    st.subheader("Upload an Image")
    uploaded = st.file_uploader("Upload", type=['jpg', 'jpeg', 'png'])
    if uploaded:
        file = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file, 1)
        show_image(img, "Uploaded Image")
        if st.button("Save Image"):
            path = save_image(img)
            st.success(f"Image saved: {path}")
    else:
        st.info("Upload an image to start.")

def Pre_Processing():
    st.subheader("processing")
    saved_images = get_saved_images()
    img = load_selected_image(saved_images, key="preprocessing_image")
    if img is not None:
        show_image(img, "Original Image")
        option = st.radio("Choose technique", [
            "Convert to GreyScale", "Median Filtering", "Gaussian Filtering",
            "Resize", "Canny_edge_detection", "Thresholding",
            "Adaptive_Thresholding", "Histogram_Equalization", "Sharpening"
        ])
        if option == "Convert to GreyScale":
            out = processing.Con_to_grey(img)
        elif option == "Median Filtering":
            out = processing.Median_Filter(img)
        elif option == "Gaussian Filtering":
            out = processing.Gaussian_Filter(img)
        elif option == "Resize":
            out = processing.Resize_Image(img)
        elif option == "Canny_edge_detection":
            out = processing.Canny_edge_detection(img)
        elif option == "Thresholding":
            out = processing.Thresholding(img)
        elif option == "Adaptive_Thresholding":
            out = processing.Adaptive_Thresholding(img)
        elif option == "Histogram_Equalization":
            out = processing.Histogram_Equalization(img)
        elif option == "Sharpening":
            out = processing.Sharpening(img)
        st.image(out, use_column_width=True)
        create_download_button(out, f"{option.lower().replace(' ', '_')}.jpg", "Download")

def segmentation():
    st.subheader("Segmentation")
    saved_images = get_saved_images()
    img = load_selected_image(saved_images, key="segmentation_image")
    if img is not None:
        show_image(img, "Original Image")
        option = st.radio("Segmentation Method", ["Threshold", "Sobel Filter", "Canny Filter"])
        if option == "Threshold":
            out = Segmentation.Thresholding(img)
        elif option == "Sobel Filter":
            out = Segmentation.Sobel(img)
        elif option == "Canny Filter":
            out = Segmentation.Canny(img)
        st.image(out, use_column_width=True)
        create_download_button(out, f"{option.lower().replace(' ', '_')}.jpg", "Download")

def main():
    st.set_page_config(page_title="Image Processing App", layout="wide")
    st.title("🧠 Image Processing Toolkit")
    menu = st.sidebar.selectbox("Choose a Task", [
        "Image Acquisition", "Pre-Processing", "Enhancement", "Segmentation", "Feature Extraction"
    ])

    ensure_directories()
    if menu == "Image Acquisition":
        image_acquisition()
    elif menu == "Pre-Processing":
        Pre_Processing()
        segmentation()

if __name__ == "__main__":
    main()
