import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

st.set_page_config(page_title="Advanced OCR App", layout="centered")

st.title("📄 Advanced Text Detection & OCR")
st.write("Upload image → detect text → extract → download")

# Language selection
lang = st.selectbox("🌐 Select Language", ["eng", "hin", "fra", "spa"])

# Upload image
uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.subheader("📷 Original Image")
    st.image(image, use_column_width=True)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    st.subheader("🧹 Processed Image")
    st.image(thresh, use_column_width=True)

    # OCR with bounding boxes
    data = pytesseract.image_to_data(thresh, lang=lang, output_type=pytesseract.Output.DICT)

    n_boxes = len(data['text'])
    extracted_text = ""

    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:
            (x, y, w, h) = (data['left'][i], data['top'][i],
                            data['width'][i], data['height'][i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            extracted_text += data['text'][i] + " "

    st.subheader("📦 Text Detection (Bounding Boxes)")
    st.image(img, use_column_width=True)

    st.subheader("🔠 Extracted Text")
    st.text_area("Output", extracted_text, height=200)

    # Download button
    st.download_button(
        label="📥 Download Text",
        data=extracted_text,
        file_name="extracted_text.txt",
        mime="text/plain"
    )
