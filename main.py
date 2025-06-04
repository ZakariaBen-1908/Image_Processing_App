import os
import cv2
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import processing
import Segmentation
import Features

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

def create_download_button(image, filename, label, original_shape=None):
    # Resize to original shape if needed
    if original_shape is not None and image.shape[:2] != original_shape[:2]:
        image = cv2.resize(image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_AREA)

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
    width = st.sidebar.slider("Image width", 100, 1000, 400)
    height = st.sidebar.slider("Image height", 100, 1000, 300)
    resized = cv2.resize(img, (width, height))
    st.image(resized, channels='BGR', caption=caption)

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
    st.subheader("Processing")
    saved_images = get_saved_images()
    img = load_selected_image(saved_images, key="preprocessing_image")

    if img is not None:
        option = st.selectbox("Choose technique", [
            "Convert to GreyScale", "Median Filtering", "Gaussian Filtering",
            "Resize", "Histogram_Equalization", "NLM Denoising", "Sharpening"
        ])
        
        if option == "Convert to GreyScale":
            out = processing.Con_to_grey(img)
        elif option == "Median Filtering":
            out = processing.Median_Filter(img)
        elif option == "Gaussian Filtering":
            out = processing.Gaussian_Filter(img)
        elif option == "Resize":
            out = processing.Resize_Image(img)
        elif option == "Histogram_Equalization":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            out = processing.Histogram_Equalization(img)

            # Plot histogram of original gray image
            fig1, ax1 = plt.subplots()
            ax1.hist(gray.ravel(), bins=256, range=[0, 256])
            ax1.set_title("Histogram of Original Gray Image")

            # Plot histogram of equalized image
            equalized_gray = out
            fig2, ax2 = plt.subplots()
            ax2.hist(equalized_gray.ravel(), bins=256, range=[0, 256])
            ax2.set_title("Histogram of Equalized Image")
        elif option == "NLM Denoising":
            out, noise, diff = processing.NLM_Denoising(img)
            original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            # Calculate the difference image
            noise_removed = cv2.absdiff(original_gray, diff_gray)
        elif option == "Sharpening":
            out = processing.Sharpening(img)

        # Display side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, channels='BGR', caption="Original Image")
        with col2:
            if len(out.shape) == 3:
                st.image(out, channels='BGR', caption=f"Processed: {option}")
            else:
                st.image(out, caption=f"Processed: {option}")
        # If NLM Denoising was selected, show the noise removed image
        if option == "Histogram_Equalization":
            col3, col4 = st.columns(2)
            with col3:
                st.pyplot(fig1)
            with col4:
                st.pyplot(fig2)
        if option == "NLM Denoising":
            col3, col4 = st.columns(2)
            with col3:
                st.image(noise, caption="Noise Removed (Difference Image)")
            with col4:
                st.image(diff, channels='BGR', caption="diff Image"),
        # Download button
        create_download_button(out, f"{option.lower().replace(' ', '_')}.jpg", "Download", original_shape=img.shape)

def segmentation():
    st.subheader("Segmentation")
    saved_images = get_saved_images()
    img = load_selected_image(saved_images, key="segmentation_image")

    if img is not None:
        option = st.selectbox("Segmentation Method", [
            "Threshold", "Adaptive_Threshold", "Sobel Filter", "Canny Filter",
            "Watershed_Segmentation", "Kmeans_Segmentation"
        ])

        if option == "Threshold":
            out = Segmentation.Thresholding(img)
        elif option == "Adaptive_Threshold":
            out = Segmentation.Adaptive_Thresholding(img)
        elif option == "Sobel Filter":
            out = Segmentation.Sobel(img)
        elif option == "Canny Filter":
            out = Segmentation.Canny(img)
        elif option == "Watershed_Segmentation":
            out = Segmentation.Watershed_Segmentation(img)
        elif option == "Kmeans_Segmentation":
            out = Segmentation.Kmeans_Segmentation(img)

        # Resize output to match input
        if out.shape[:2] != img.shape[:2]:
            out = cv2.resize(out, (img.shape[1], img.shape[0]))

        # Side-by-side display
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, channels='BGR', caption="Original Image")
        with col2:
            if len(out.shape) == 3:
                st.image(out, channels='BGR', caption=f"Processed: {option}")
            else:
                st.image(out, caption=f"Processed: {option}")

        create_download_button(out, f"{option.lower().replace(' ', '_')}.jpg", "Download", original_shape=img.shape)

def feature_extraction():
    st.subheader("Feature Extraction")
    saved_images = get_saved_images()
    img = load_selected_image(saved_images)
    if img is not None:
        show_image(img, "Original Image")
        option = st.radio("Feature Method", ["Mean", "Standard Deviation", "Text Extractor"])
        if option == "Mean":
            st.write(f"Mean pixel value: {Features.Mean(img)}")
        elif option == "Standard Deviation":
            st.write(f"Standard deviation: {Features.Std_deviation(img)}")
        elif option == "Text Extractor":  
            extracted_text = Features.extract_text(img)
            st.text_area("Extracted Text", extracted_text, height=300)


def main():
    st.set_page_config(page_title="Image Processing App", layout="wide")
    st.title("🧠 Image Processing Toolkit")
    menu = st.sidebar.selectbox("Choose a Task", [
        "Image Acquisition", "Pre-Processing", "Segmentation", "Feature Extraction"
    ])

    ensure_directories()
    if menu == "Image Acquisition":
        image_acquisition()
    elif menu == "Pre-Processing":
        Pre_Processing()
    elif menu == "Segmentation":
        segmentation()
    elif menu == "Feature Extraction":
        feature_extraction()

if __name__ == "__main__":
    main()
