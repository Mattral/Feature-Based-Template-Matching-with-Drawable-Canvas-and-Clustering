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
        cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=(0, 255, 0), thickness=2)
    st.image(image, caption=title, use_column_width=True)

def feature_match_and_box(image, template):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(image, None)

    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    st.write(f"Number of good matches: {len(good_matches)}")

    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if matrix is not None:
            # Draw a rectangle around the matched region
            h, w = template.shape[:2]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            bounding_box = cv2.boundingRect(dst)
            st.write(f"Bounding box: {bounding_box}")
            cv2.rectangle(image, (int(bounding_box[0]), int(bounding_box[1])), 
                          (int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3])), 
                          (0, 255, 0), 2)
        else:
            st.error("No homography found. Check the quality of matches.")
    else:
        st.error("Not enough matches are found - {}/{}".format(len(good_matches), 4))

    return image


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
                    matched_img = feature_match_and_box(img, cropped_template)
                    display_image(matched_img, "Image with Matched Area")
    else:
        st.warning("Please upload both images to proceed.")

if __name__ == "__main__":
    main()
