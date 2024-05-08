import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
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

def apply_sift_and_cluster(img, template, lowe_ratio):
    """Apply SIFT matching and clustering to find and highlight multiple instances of a template."""
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(template, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = [m for m, n in matches if m.distance < lowe_ratio * n.distance]

    # Extract location of good matches
    points = np.zeros((len(good_matches), 2), dtype=np.float32)
    for i, match in enumerate(good_matches):
        points[i, :] = keypoints1[match.queryIdx].pt

    # Clustering of matched points
    kmeans = KMeans(n_clusters=min(10, len(good_matches)//5), random_state=0).fit(points)
    labels = kmeans.labels_

    match_img = cv2.drawMatches(img, keypoints1, template, keypoints2, good_matches, None)
    box_img = img.copy()

    # Draw clusters
    for i in range(kmeans.n_clusters):
        cluster_points = points[labels == i]
        if len(cluster_points) > 3:
            x, y, w, h = cv2.boundingRect(cluster_points)
            cv2.rectangle(box_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return match_img, box_img

def main():
    st.title("Feature-Based Template Matching with Drawable Canvas and Clustering")

    img_file = st.sidebar.file_uploader("Upload your Image", type=["png", "jpg", "jpeg"])
    template_file = st.sidebar.file_uploader("Upload your Template Image", type=["png", "jpg", "jpeg"])
    lowe_ratio = st.sidebar.slider('Adjust Lowe Ratio', min_value=0.0, max_value=1.0, value=0.75, step=0.05,
                                   help="Lower values of the ratio are more strict, reducing false positives but may miss some correct matches. Higher values increase the number of matches but may include more incorrect ones.")

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
                    matched_img, box_img = apply_sift_and_cluster(img.copy(), cropped_template, lowe_ratio)
                    st.subheader("Image with Match Points")
                    display_image(matched_img, "Match Points Image")
                    st.subheader("Image with Matched Area Box")
                    display_image(box_img, "Bounding Box Image")

    else:
        st.warning("Please upload both images to proceed.")

if __name__ == "__main__":
    main()
