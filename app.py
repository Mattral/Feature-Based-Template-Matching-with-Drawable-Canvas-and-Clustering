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
    st.image(image, caption=title, use_column_width=True)

def apply_sift_matching(img, template, lowe_ratio=0.75):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(template, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < lowe_ratio * n.distance]

    if len(good_matches) > 4:
        result_img = cv2.drawMatches(img, keypoints1, template, keypoints2, good_matches, None)
        return result_img
    else:
        st.error("Not enough matches were found - try adjusting the parameters or selecting a different template region.")
        return img

def main():
    st.title("Feature-Based Template Matching with Drawable Canvas")

    img_file = st.sidebar.file_uploader("Upload your Image", type=["png", "jpg", "jpeg"])
    template_file = st.sidebar.file_uploader("Upload your Template Image", type=["png", "jpg", "jpeg"])
    lowe_ratio = st.sidebar.slider("Lowe's ratio test threshold", 0.4, 0.9, 0.75, 0.05)

    if img_file and template_file:
        img = load_image(img_file)
        template = load_image(template_file)

        display_image(img, "Uploaded Image")
        display_image(template, "Uploaded Template")

        st.subheader("Draw cropping area on the template:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)", stroke_width=2,
            stroke_color="#FFFFFF", background_image=Image.fromarray(template),
            update_streamlit=True, drawing_mode="rect", key="canvas",
        )

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if objects:
                rect = objects[0]
                x, y = int(rect['left']), int(rect['top'])
                width, height = int(rect['width']), int(rect['height'])
                cropped_template = template[y:y+height, x:x+width]
                display_image(cropped_template, "Cropped Template")

                if st.button("Match Template"):
                    with st.spinner("Matching..."):
                        result_img = apply_sift_matching(img, cropped_template, lowe_ratio)
                        display_image(result_img, "Image with Matched Areas")
    else:
        st.warning("Please upload both images to proceed.")

if __name__ == "__main__":
    main()
