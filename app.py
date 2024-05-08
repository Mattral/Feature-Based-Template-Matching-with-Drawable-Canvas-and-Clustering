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
    st.image(image, caption=title, use_column_width=True)

def apply_sift_matching(img, template, lowe_ratio=0.75):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(template, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = [m for m, n in matches if m.distance < lowe_ratio * n.distance]

    st.write(f"Total matches found: {len(matches)}")
    st.write(f"Good matches using Lowe's ratio: {len(good_matches)}")

    match_img = cv2.drawMatches(img, keypoints1, template, keypoints2, good_matches, None)

    box_img = img.copy()
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if matrix is not None:
            h, w = template.shape[:2]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            box_img = cv2.polylines(box_img, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            st.write("Homography could not be computed successfully.")
    else:
        st.write("Not enough good matches are found - {}/{}".format(len(good_matches), 5))

    return match_img, box_img

def main():
    st.title("Feature-Based Template Matching with Drawable Canvas")

    img_file = st.sidebar.file_uploader("Upload your Image", type=["png", "jpg", "jpeg"])
    template_file = st.sidebar.file_uploader("Upload your Template Image", type=["png", "jpg", "jpeg"])

    if img_file and template_file:
        img = load_image(img_file)
        template = load_image(template_file)

        col1, col2 = st.columns(2)
        with col1:
            display_image(img, "Uploaded Image")
        with col2:
            display_image(template, "Uploaded Template")

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
                display_image(cropped_template, "Cropped Template")

                if st.button("Match Template"):
                    matched_img, box_img = apply_sift_matching(img.copy(), cropped_template)
                    st.subheader("Image with Match Points")
                    display_image(matched_img, "Match Points Image")
                    st.subheader("Image with Matched Area Box")
                    display_image(box_img, "Bounding Box Image")

    else:
        st.warning("Please upload both images to proceed.")

if __name__ == "__main__":
    main()
