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

def non_max_suppression_fast(boxes, overlapThresh):
    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # Initialize the list of picked indexes	
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]

    # Compute the area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list, add the index value to the list of picked indexes, then initialize the suppression list (i.e., indexes that will be deleted)
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # Loop over all indexes in the indexes list
        for pos in range(0, last):
            # Grab the current index
            j = idxs[pos]

            # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # Compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # Compute the ratio of overlap between the computed bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # If there is sufficient overlap, suppress the current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # Delete all indexes from the index list that are in the suppression list
        idxs = np.delete(idxs, suppress)

    # Return only the bounding boxes that were picked
    return boxes[pick]

def match_template(img, template):
    """Match template and highlight matching areas on the image."""
    method = cv2.TM_CCOEFF_NORMED
    res = cv2.matchTemplate(img, template, method)
    threshold = 0.8
    loc = np.where(res >= threshold)
    w, h = template.shape[::-1]

    # Collect all potential boxes
    boxes = []
    for pt in zip(*loc[::-1]):  # Switch x and y coordinates
        boxes.append([int(pt[0]), int(pt[1]), int(w), int(h)])

    # Convert boxes to a numpy array for NMS
    boxes_np = np.array(boxes)
    # Apply non-maximum suppression
    picked_boxes = non_max_suppression_fast(boxes_np, 0.3)  # Overlap threshold can be adjusted

    for (startX, startY, width, height) in picked_boxes:
        cv2.rectangle(img, (startX, startY), (startX + width, startY + height), (0, 255, 0), 2)

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
