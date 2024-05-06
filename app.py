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

def feature_match_and_box(image, template):
    sift = cv2.SIFT_create()  # Use SIFT instead of ORB
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(image, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if matrix is not None:
            h, w = template.shape[:2]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            bounding_box = cv2.boundingRect(dst)
            cv2.rectangle(image, (int(bounding_box[0]), int(bounding_box[1])), 
                          (int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3])), 
                          (0, 255, 0), 2)
        else:
            st.error("No homography found. Check the quality of matches.")
    else:
        st.error(f"Not enough matches are found - {len(good_matches)}/4")

    return image

def main():
    st.title("Feature Matching with SIFT and Cropping Tool")
    img_file = st.sidebar.file_uploader("Upload your Image", type=["png", "jpg", "jpeg"])
    template_file = st.sidebar.file_uploader("Upload your Template Image", type=["png", "jpg", "jpeg"])

    if img_file and template_file:
        img = load_image(img_file)
        template = load_image(template_file)

        st.image(template, caption="Uploaded Template", use_column_width=True)
        
        # Setup canvas for user cropping
        st.subheader("Draw cropping area on the template:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Use a transparent fill color
            stroke_width=2,
            stroke_color="#FFFFFF",
            background_image=Image.open(template_file).convert('RGB'),
            update_streamlit=True,
            height=template.shape[0],
            width=template.shape[1],
            drawing_mode="rect",
            key="canvas",
        )

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            if objects:
                rect = objects[0]
                x = int(rect['left'])
                y = int(rect['top'])
                width = int(rect['width'])
                height = int(rect['height'])
                cropped_template = template[y:y + height, x:x + width]
                st.image(cropped_template, caption="Cropped Template", use_column_width=True)

                if st.button("Match Template"):
                    result_image = feature_match_and_box(img, cropped_template)
                    st.image(result_image, caption="Matched Image", use_column_width=True)
    else:
        st.warning("Please upload both images to proceed.")

if __name__ == "__main__":
    main()
