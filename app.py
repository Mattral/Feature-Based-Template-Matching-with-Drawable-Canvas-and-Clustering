import streamlit as st
import cv2
import numpy as np
from PIL import Image

def load_image(image_file):
    """Converts the uploaded file to an OpenCV image."""
    image = Image.open(image_file)
    image = np.array(image)
    if image.shape[-1] == 4:  # Convert RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image is cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def template_crop(template):
    """Crop the template image based on some logic."""
    # This is a dummy crop; you might want to replace it with actual coordinates
    cropped = template[10:100, 10:100]  # Crop a portion of the image
    return cropped

def invariantMatchTemplate(image, template, method_name, rot_range, scale_range):
    """Perform matching with rotation and scaling invariant."""
    method = eval(f"cv2.{method_name}")  # Convert the method name to a cv2 method
    best_match = None
    best_score = -1
    for angle in np.arange(rot_range[0], rot_range[1], rot_range[2]):
        for scale in np.arange(scale_range[0], scale_range[1], scale_range[2]):
            scaled_template = cv2.resize(template, None, fx=scale/100, fy=scale/100, interpolation=cv2.INTER_AREA)
            rotated_template = Image.fromarray(scaled_template)
            rotated_template = rotated_template.rotate(angle, expand=True)
            rotated_template = np.array(rotated_template)

            if rotated_template.shape[0] > image.shape[0] or rotated_template.shape[1] > image.shape[1]:
                continue

            result = cv2.matchTemplate(image, rotated_template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                score = -min_val
                location = min_loc
            else:
                score = max_val
                location = max_loc

            if score > best_score:
                best_score = score
                best_match = location

    return best_match

def main():
    st.title("Template Matching App")

    img_file = st.sidebar.file_uploader("Upload your Image", type=["png", "jpg", "jpeg"])
    template_file = st.sidebar.file_uploader("Upload your Template Image", type=["png", "jpg", "jpeg"])

    if img_file and template_file:
        img = load_image(img_file)
        template = load_image(template_file)

        if st.button("Match Template"):
            cropped_template = template_crop(template)
            if cropped_template is not None and cropped_template.size > 0:
                method_name = "TM_CCOEFF_NORMED"
                rot_range = [0, 360, 10]  # Degrees
                scale_range = [100, 150, 10]  # Percentage
                best_match = invariantMatchTemplate(img, cropped_template, method_name, rot_range, scale_range)
                if best_match:
                    st.write("Best match at location:", best_match)
                else:
                    st.error("No suitable match found.")
            else:
                st.error("Cropping returned an empty image or failed.")
    else:
        st.warning("Please upload both images to proceed.")

if __name__ == "__main__":
    main()
