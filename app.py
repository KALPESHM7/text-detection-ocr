import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

# (Optional) Set Tesseract path for Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="Text Detection OCR", layout="centered")

st.title("📄 Text Detection and Extraction (OCR)")
st.write("Upload an image to extract text using OpenCV + Tesseract OCR")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.subheader("📷 Original Image")
    st.image(image, use_column_width=True)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    st.subheader("🧹 Processed Image")
    st.image(thresh, use_column_width=True)

    # OCR
    text = pytesseract.image_to_string(thresh)

    st.subheader("🔠 Extracted Text")
    st.text_area("Output", text, height=200)