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

def rotate_image(image, angle):
    """Rotate an image."""
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def match_template(img, template, max_angle=360, step=45):
    """Match template with rotation and highlight all matching areas."""
    best_matches = []
    method = cv2.TM_CCOEFF_NORMED
    for angle in range(0, max_angle, step):
        rotated_template = rotate_image(template, angle)
        res = cv2.matchTemplate(img, rotated_template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # Store the best matches and their positions
        loc = np.where(res >= 0.7)  # Threshold for detecting matches
        for pt in zip(*loc[::-1]):  # Iterate over matches
            if not best_matches or all(np.linalg.norm(np.array(pt) - np.array(m['position'])) > 10 for m in best_matches):
                best_matches.append({'position': pt, 'angle': angle, 'value': res[pt[::-1]]})

    # Draw all matches
    for match in best_matches:
        pt = match['position']
        cv2.rectangle(img, pt, (pt[0] + rotated_template.shape[1], pt[1] + rotated_template.shape[0]), (0, 255, 0), 2)

    return img

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
        canvas_width, canvas_height = template.shape[1], template.shape[0]
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Transparent fill color
            stroke_width=2,
            stroke_color="#FFFFFF",
            background_image=Image.fromarray(template),
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
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
                display_image(cropped_template, "Cropped Template")

                if st.button("Match Template"):
                    result_img = match_template(img.copy(), cropped_template)  # Use copy of image for drawing
                    display_image(result_img, "Image with Matched Areas")
    else:
        st.warning("Please upload both images to proceed.")

if __name__ == "__main__":
    main()
