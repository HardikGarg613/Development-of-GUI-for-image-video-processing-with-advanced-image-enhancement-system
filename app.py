import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Image Enhancement System", layout="wide")

st.title("Advanced Image & Video Processing System")
st.write("Upload an image and apply enhancement techniques.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

technique = st.selectbox(
    "Select Enhancement Technique",
    [
        "Histogram Equalization",
        "CLAHE",
        "Gamma Correction",
        "Edge Enhancement",
        "Noise Reduction"
    ]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.subheader("Original Image")
    st.image(img, use_container_width=True)

    processed = img.copy()

    if technique == "Histogram Equalization":

        gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        processed = cv2.equalizeHist(gray)

    elif technique == "CLAHE":

        gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        processed = clahe.apply(gray)

    elif technique == "Gamma Correction":

        gamma = 1.5
        invGamma = 1.0 / gamma

        table = np.array([
            ((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")

        processed = cv2.LUT(processed, table)

    elif technique == "Edge Enhancement":

        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])

        processed = cv2.filter2D(processed, -1, kernel)

    elif technique == "Noise Reduction":

        processed = cv2.GaussianBlur(processed, (5, 5), 0)

    st.subheader("Processed Image")
    st.image(processed, use_container_width=True)
