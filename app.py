import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

def load_image(image_file):
    """Converts the uploaded file to an OpenCV image."""
    image = Image.open(image_file).convert("RGB")  # Ensure image is in RGB
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    return image

def display_image(image, title, box=None):
    """Display an image with a caption and optional bounding box."""
    if box:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    st.image(image, caption=title, use_column_width=True)

def rotate_image(image, angle):
    """Rotate an image by an angle while preserving the entire content."""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def invariant_match_template(image, template, method=cv2.TM_CCOEFF_NORMED, angle_step=10, scale_step=10, threshold=0.8):
    """Match template considering rotation and scale invariance."""
    detected_boxes = []
    for angle in range(0, 360, angle_step):
        rotated_template = rotate_image(template, angle)
        for scale in np.linspace(0.5, 1.5, num=int((1.5-0.5)/(scale_step/100.0)+1)):
            scaled_template = cv2.resize(rotated_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scaled_template.shape[0] > image.shape[0] or scaled_template.shape[1] > image.shape[1]:
                continue
            res = cv2.matchTemplate(image, scaled_template, method)
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                detected_boxes.append([pt[0], pt[1], pt[0] + scaled_template.shape[1], pt[1] + scaled_template.shape[0]])

    return np.array(detected_boxes)

def main():
    st.title("Template Matching with Drawable Canvas")

    img_file = st.sidebar.file_uploader("Upload your Image", type=["png", "jpg", "jpeg"])
    template_file = st.sidebar.file_uploader("Upload your Template Image", type=["png", "jpg", "jpeg"])

    if img_file and template_file:
        img = load_image(img_file)
        template = load_image(template_file)

        display_image(img, "Uploaded Image")
        display_image(template, "Uploaded Template")

        # Setup canvas for user cropping
        st.subheader("Draw cropping area on the template:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Transparent fill color
            stroke_width=2,
            stroke_color="#FFFFFF",
            background_image=Image.fromarray(cv2.cvtColor(template, cv2.COLOR_BGR2RGB)),
            update_streamlit=True,
            drawing_mode="rect",
            key="canvas"
        )

        if canvas_result.json_data and "objects" in canvas_result.json_data["objects"]:
            rect = canvas_result.json_data["objects"][0]
            x, y, width, height = int(rect['left']), int(rect['top']), int(rect['width']), int(rect['height'])
            cropped_template = template[y:y+height, x:x+width]
            display_image(cropped_template, "Cropped Template")

            if st.button("Match Template"):
                detected_boxes = invariant_match_template(img, cropped_template)
                for box in detected_boxes:
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                display_image(img, "Image with Matched Areas")
        else:
            st.warning("No cropping area selected. Please draw a rectangle on the template.")
    else:
        st.warning("Please upload both images to proceed.")

if __name__ == "__main__":
    main()
