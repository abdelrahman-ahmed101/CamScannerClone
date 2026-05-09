import streamlit as st
import numpy as np
import cv2
from PIL import Image

from utils.processing import process_document
from utils.image_utils import convert_to_display


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="CamScanner CV Project",
    layout="wide",
)


# =========================
# Title
# =========================

st.title("CamScanner using Computer Vision")
st.write("Upload or capture an image and watch every processing step.")


# =========================
# Upload Section
# =========================

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
)

camera_image = st.camera_input("Capture Document")


# =========================
# Select Source
# =========================

selected_file = None

if uploaded_file is not None:
    selected_file = uploaded_file

elif camera_image is not None:
    selected_file = camera_image


# =========================
# Process Image
# =========================

if selected_file is not None:

    file_bytes = np.asarray(
        bytearray(selected_file.read()),
        dtype=np.uint8,
    )

    image = cv2.imdecode(file_bytes, 1)

    st.subheader("Uploaded Image")
    st.image(convert_to_display(image), use_container_width=True)

    if st.button("Start Processing"):

        with st.spinner("Processing document..."):

            results = process_document(image)

        st.divider()

        st.header("Processing Steps")
        # =========================
        # Show All Steps
        # =========================

        for index, (title, step_image) in enumerate(results["steps"]):

            st.subheader(f"Step {index + 1}: {title}")

            st.image(
                convert_to_display(step_image),
                use_container_width=True,
            )

        # =========================
        # OCR Result
        # =========================

        if results["success"]:
            st.divider()

            st.header("Final Scanned Document")

            st.image(
                convert_to_display(results["final_image"]),
                use_container_width=True,
            )

            st.header("Extracted Text")

            st.text_area(
                "OCR Output",
                results["text"],
                height=300,
            )

        else:
            st.error(results["message"])
