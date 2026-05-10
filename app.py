# app.py

import streamlit as st
import tempfile
import os
import cv2
from main import restore_document, restore_document_simple

st.set_page_config(
    page_title="Document Restoration",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Restoration App")
st.write("Upload a document image and restore/enhance it using OpenCV.")

# Upload image
uploaded_file = st.file_uploader(
    "Upload a document image",
    type=["png", "jpg", "jpeg"]
)

# Processing option
mode = st.radio(
    "Choose Processing Mode",
    ["Full Pipeline (with steps)", "Simple Pipeline"]
)

if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    st.subheader("Original Image")
    st.image(temp_path, use_container_width=True)

    if st.button("Restore Document"):

        with st.spinner("Processing image..."):

            if mode == "Full Pipeline (with steps)":
                output = restore_document(temp_path, show_steps=False)
            else:
                output = restore_document_simple(temp_path)

        if output is not None:

            st.success("Document restored successfully!")

            # Output file path
            base_path = os.path.splitext(temp_path)[0]
            output_path = f"{base_path}_restored.png"

            # Show restored image
            st.subheader("Restored Image")
            st.image(output_path, use_container_width=True)

            # Download button
            with open(output_path, "rb") as file:
                st.download_button(
                    label="⬇ Download Restored Image",
                    data=file,
                    file_name="restored_document.png",
                    mime="image/png"
                )

        else:
            st.error("Failed to process image.")