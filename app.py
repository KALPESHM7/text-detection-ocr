import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

st.set_page_config(page_title="Multi-Language OCR", layout="centered")

st.title("Smart Document Scanner & OCR")
st.write("Auto-detect language + extract text + download")

# Mode selection
mode = st.radio("Select Mode", ["Auto Detect", "Manual Select"])

# Manual language option
if mode == "Manual Select":
    lang = st.selectbox("🌐 Select Language", ["eng", "hin", "tam", "kan"])
else:
    lang = "eng+hin+tam+kan"   # multi-language auto detection

# Upload image
uploaded_file = st.file_uploader("📤 Upload Document", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.subheader("📷 Original Image")
    st.image(image, use_column_width=True)

    # --- Document Scanner Effect ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    st.subheader("📄 Scanned Document")
    st.image(thresh, use_column_width=True)

    # --- OCR with Bounding Boxes ---
    data = pytesseract.image_to_data(
        thresh,
        lang=lang,
        config='--psm 6',
        output_type=pytesseract.Output.DICT
    )

    n_boxes = len(data['text'])
    extracted_text = ""

    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:
            (x, y, w, h) = (
                data['left'][i],
                data['top'][i],
                data['width'][i],
                data['height'][i]
            )
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            extracted_text += data['text'][i] + " "

    st.subheader("📦 Detected Text Regions")
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
