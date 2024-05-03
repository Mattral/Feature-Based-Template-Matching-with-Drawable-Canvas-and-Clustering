import streamlit as st
import cv2
import numpy as np
from PIL import Image
import Modified  # Assuming Modified.py contains all the necessary functions

def load_image(image_file):
    """Converts the uploaded file to an OpenCV image."""
    image = Image.open(image_file)
    image = np.array(image)
    if image.shape[-1] == 4:  # Convert RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def main():
    st.title("Template Matching App")

    img_file = st.sidebar.file_uploader("Upload your Image", type=["png", "jpg", "jpeg"])
    template_file = st.sidebar.file_uploader("Upload your Template Image", type=["png", "jpg", "jpeg"])

    if img_file and template_file:
        img = load_image(img_file)
        template = load_image(template_file)

        if st.button("Match Template"):
            # Use your function from Modified.py
            cropped_template = Modified.template_crop(template)
            if cropped_template is not None and cropped_template.size > 0:
                method_name = "TM_CCOEFF_NORMED"
                rot_range = [0, 360, 10]  # Start, end, step
                scale_range = [100, 150, 10]  # Start, end, step
                points_list = Modified.invariantMatchTemplate(img, cropped_template, method_name, rot_range, scale_range)
                # Display results or further processing
                st.write("Match found at points:", points_list)
            else:
                st.error("Cropping returned an empty image or failed.")
    else:
        st.warning("Please upload both images to proceed.")

if __name__ == "__main__":
    main()
