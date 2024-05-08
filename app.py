import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

def load_image(image_file):
    """Converts the uploaded file to an OpenCV image."""
    image = Image.open(image_file)
    image = np.array(image)
    if image.shape[-1] == 4:  # Convert RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def display_image(image, title, box=None):
    """Display an image with a caption and optional bounding box."""
    if box:
        # Draw rectangle on the image
        cv2.rectangle(image, box[0], box[1], color=(0, 255, 0), thickness=2)
    st.image(image, caption=title, use_column_width=True)

def match_template(img, template):
    """Match template and highlight matching areas on the image."""
    method = cv2.TM_CCOEFF_NORMED
    res = cv2.matchTemplate(img, template, method)
    threshold = 0.6
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):  # Switch x and y coordinates
        cv2.rectangle(img, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 0), 2)
    return img

def main():
    st.title("Interactive Template Matching")

    # Upload sections in the sidebar
    st.sidebar.header("Upload Images")
    img_file = st.sidebar.file_uploader("Upload your Image", type=["png", "jpg", "jpeg"])
    template_file = st.sidebar.file_uploader("Upload your Template Image", type=["png", "jpg", "jpeg"])

    if img_file and template_file:
        img = load_image(img_file)
        template = load_image(template_file)

        st.subheader("Uploaded Images")
        col1, col2 = st.columns(2)
        with col1:
            display_image(img, "Original Image")
        with col2:
            display_image(template, "Template Image")

        # Setup canvas for user cropping
        st.subheader("Draw cropping area on the template:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Transparent fill color
            stroke_width=2,
            stroke_color="#FFFFFF",
            background_image=Image.fromarray(template),
            update_streamlit=True,
            height=template.shape[0],
            width=template.shape[1],
            drawing_mode="rect",
            key="canvas",
        )

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if objects:
                rect = objects[0]
                x = int(rect['left'])
                y = int(rect['top'])
                width = int(rect['width'])
                height = int(rect['height'])
                cropped_template = template[y:y + height, x:x + width]
                st.subheader("Cropped Template")
                display_image(cropped_template, "Cropped Template for Matching")

                if st.button("Match Template"):
                    st.subheader("Matched Result")
                    result_img = match_template(img.copy(), cropped_template)
                    display_image(result_img, "Image with Matched Areas")
    else:
        st.warning("Please upload both images to proceed.")

if __name__ == "__main__":
    main()
