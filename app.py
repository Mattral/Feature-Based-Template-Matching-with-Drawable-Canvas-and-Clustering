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
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important as we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes    
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")



def rotate_image(image, angle):
    """Rotate an image by an angle while preserving the entire content."""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def invariant_match_template(image, template, method=cv2.TM_CCOEFF_NORMED, angle_step=10, scale_step=10, threshold=0.8):
    """Match template considering rotation and scale invariance."""
    detected_boxes = []
    for angle in np.arange(0, 360, angle_step):
        rotated_template = rotate_image(template, angle)
        for scale in np.arange(0.5, 1.5, scale_step / 100.0):
            scaled_template = cv2.resize(rotated_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scaled_template.shape[0] > image.shape[0] or scaled_template.shape[1] > image.shape[1]:
                continue
            res = cv2.matchTemplate(image, scaled_template, method)
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):  # Switch columns and rows
                detected_boxes.append([pt[0], pt[1], pt[0] + scaled_template.shape[1], pt[1] + scaled_template.shape[0]])

    return np.array(detected_boxes)



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
                x, y, width, height = adjust_canvas_to_image(rect, canvas_width, canvas_height, template.shape)
                cropped_template = template[y:y+height, x:x+width]
                display_image(cropped_template, "Cropped Template")

                if st.button("Match Template"):
                    display_img = img.copy()  # Make a deep copy to draw boxes on
                    matches = invariant_match_template(display_img, cropped_template)
                    filtered_matches = non_max_suppression_fast(matches, 0.3)  # Adjust threshold as needed

                    if len(filtered_matches) > 0:
                        for (x1, y1, x2, y2) in filtered_matches:
                            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        display_image(display_img, "Image with Matched Areas")
                        st.write(f"Found {len(filtered_matches)} matches")
                    else:
                        st.error("No suitable matches found.")
    else:
        st.warning("Please upload both images to proceed.")

def adjust_canvas_to_image(rect, canvas_width, canvas_height, shape):
    scale_x = shape[1] / canvas_width
    scale_y = shape[0] / canvas_height
    x = int(rect['left'] * scale_x)
    y = int(rect['top'] * scale_y)
    width = int(rect['width'] * scale_x)
    height = int(rect['height'] * scale_y)
    return x, y, width, height

# The rest of your function implementations (load_image, display_image, invariant_match_template, etc.) will be the same as provided earlier.

if __name__ == "__main__":
    main()
