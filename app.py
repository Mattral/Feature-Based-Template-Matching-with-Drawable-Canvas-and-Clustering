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

    """Display an image with a caption and optional bounding box."""

def display_image(image, title, box=None):
    if box:
        # Draw rectangle on the image
        cv2.rectangle(image, box[0], box[1], color=(0, 255, 0), thickness=2)
    st.image(image, caption=title, use_column_width=True)
    
def invariantMatchTemplate(image, template, method_name, rot_range, scale_range):
    method = eval(f"cv2.{method_name}")
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

        display_image(img, "Uploaded Image")
        display_image(template, "Uploaded Template")

        # Setup canvas for user cropping
        st.subheader("Draw cropping area on the template:")
        canvas_width, canvas_height = 600, 500  # Match these to the 'width' and 'height' parameters in st_canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Use a transparent fill color
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
                # Assuming the first object is the rectangle
                rect = objects[0]
                # Canvas to image scaling factors
                scale_x = template.shape[1] / canvas_width
                scale_y = template.shape[0] / canvas_height

                # Adjusted coordinates
                x = int(rect['left'] * scale_x)
                y = int(rect['top'] * scale_y)
                width = int(rect['width'] * scale_x)
                height = int(rect['height'] * scale_y)
                x_end = x + width
                y_end = y + height

                # Debugging output
                st.write(f"Adjusted Rectangle Coordinates: x={x}, y={y}, width={width}, height={height}")

                # Crop the template image according to the adjusted rectangle coordinates
                cropped_template = template[y:y_end, x:x_end]
                display_image(cropped_template, "Cropped Template")

                if st.button("Match Template"):
                    method_name = "TM_CCOEFF_NORMED"
                    rot_range = [0, 360, 10]
                    scale_range = [100, 150, 10]
                    best_match = invariantMatchTemplate(img, cropped_template, method_name, rot_range, scale_range)
                    if best_match:
                        st.write("Best match at location:", best_match)
                        match_top_left = best_match
                        match_bottom_right = (best_match[0] + width, best_match[1] + height)
                        display_image(img, "Image with Matched Area", box=(match_top_left, match_bottom_right))
                    else:
                        st.error("No suitable match found.")
    else:
        st.warning("Please upload both images to proceed.")

if __name__ == "__main__":
    main()
