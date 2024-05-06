import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

def load_image(image_file):
    image = Image.open(image_file)
    image = np.array(image)
    if image.shape[-1] == 4:  # Convert RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def display_image(image, title, box=None):
    if box:
        cv2.rectangle(image, box[0], box[1], color=(0, 255, 0), thickness=2)
    st.image(image, caption=title, use_column_width=True)

def feature_match(image, template):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(image, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 10:
        matches = matches[:10]  # Take the top 10 matches

    # Draw matches on the image
    matched_image = cv2.drawMatches(template, kp1, image, kp2, matches, None, flags=2)

    return matched_image

def main():
    st.title("Template Matching App")

    img_file = st.sidebar.file_uploader("Upload your Image", type=["png", "jpg", "jpeg"])
    template_file = st.sidebar.file_uploader("Upload your Template Image", type=["png", "jpg", "jpeg"])

    if img_file and template_file:
        img = load_image(img_file)
        template = load_image(template_file)

        display_image(img, "Uploaded Image")
        display_image(template, "Uploaded Template")

        st.subheader("Draw cropping area on the template:")
        canvas_width, canvas_height = 300, 250
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)", 
            stroke_width=2,
            stroke_color="#FFFFFF",
            background_image=Image.open(template_file).resize((canvas_width, canvas_height)),
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect",
            key="canvas",
        )

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            if objects:
                rect = objects[0]
                scale_x = template.shape[1] / canvas_width
                scale_y = template.shape[0] / canvas_height

                x = int(rect['left'] * scale_x)
                y = int(rect['top'] * scale_y)
                width = int(rect['width'] * scale_x)
                height = int(rect['height'] * scale_y)
                x_end = x + width
                y_end = y + height

                cropped_template = template[y:y_end, x:x_end]
                display_image(cropped_template, "Cropped Template")

                if st.button("Match Template"):
                    matched_img = feature_match(img, cropped_template)
                    display_image(matched_img, "Matched Image")
    else:
        st.warning("Please upload both images to proceed.")

if __name__ == "__main__":
    main()
