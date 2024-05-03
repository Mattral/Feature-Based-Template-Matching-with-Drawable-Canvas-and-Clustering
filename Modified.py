import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

box_points = []
button_down = False

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def scale_image(image, percent, maxwh):
    max_width = maxwh[1]
    max_height = maxwh[0]
    max_percent_width = max_width / image.shape[1] * 100
    max_percent_height = max_height / image.shape[0] * 100
    max_percent = 0
    if max_percent_width < max_percent_height:
        max_percent = max_percent_width
    else:
        max_percent = max_percent_height
    if percent > max_percent:
        percent = max_percent
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    result = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return result, percent

import numpy as np
import cv2

def click_and_crop(event, x, y, flags, param):
    global box_points, button_down
    # Ensure param is writable
    if not param.flags['WRITEABLE']:
        param.setflags(write=1)
    
    if (button_down == False) and (event == cv2.EVENT_LBUTTONDOWN):
        button_down = True
        box_points = [(x, y)]
    elif (button_down == True) and (event == cv2.EVENT_MOUSEMOVE):
        if len(box_points) == 1:
            image_copy = param.copy()
            point = (x, y)
            cv2.rectangle(image_copy, box_points[0], point, (0, 255, 0), 2)
            cv2.imshow("Template Cropper - Press C to Crop", image_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        button_down = False
        if len(box_points) == 1:
            box_points.append((x, y))
            cv2.rectangle(param, box_points[0], box_points[1], (0, 255, 0), 2)
            cv2.imshow("Template Cropper - Press C to Crop", param)

def template_crop(image):
    global box_points, button_down
    box_points = []
    button_down = False
    clone = image.copy()
    cv2.namedWindow("Template Cropper - Press C to Crop")
    cv2.setMouseCallback("Template Cropper - Press C to Crop", click_and_crop, clone)

    cropped_region = None
    while True:
        cv2.imshow("Template Cropper - Press C to Crop", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(box_points) == 2:
                # Correctly sort x and y coordinates
                x1, x2 = sorted([box_points[0][0], box_points[1][0]])
                y1, y2 = sorted([box_points[0][1], box_points[1][1]])
                cropped_region = clone[y1:y2, x1:x2]
            break
        elif key == 27:  # ESC
            break

    cv2.destroyAllWindows()
    return cropped_region

def calculate_template_dimensions_at_scale(template, scale_percent):
    """
    Calculate the dimensions of the template after scaling.
    Args:
    template (numpy array): The template image.
    scale_percent (float): The scaling percentage.

    Returns:
    tuple: (width, height) of the scaled template
    """
    new_width = int(template.shape[1] * scale_percent / 100)
    new_height = int(template.shape[0] * scale_percent / 100)
    return new_width, new_height


def rotate_and_scale(template, angle, scale_factor, image_maxwh):
    """Applies rotation and scaling to the template."""
    scaled_height = int(template.shape[0] * scale_factor)
    scaled_width = int(template.shape[1] * scale_factor)
    scaled_template = cv2.resize(template, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    if angle != 0:
        center = (scaled_width // 2, scaled_height // 2)
        matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated_template = cv2.warpAffine(scaled_template, matrix, (scaled_width, scaled_height))
    else:
        rotated_template = scaled_template

    return rotated_template

def match_template(image, template, method_info):
    # Ensure the correct integer method constant is used for matching
    cv2_method = method_info["cv2_method"]
    result = cv2.matchTemplate(image, template, cv2_method)  # use the integer constant
    threshold = method_info["threshold"]
    if cv2_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        loc = np.where(result <= threshold)
    else:
        loc = np.where(result >= threshold)

    return list(zip(*loc[::-1]))



# Configuration dictionary for methods
methods = {
    "TM_CCOEFF": {
        "cv2_method": cv2.TM_CCOEFF,
        "threshold": 0.8  # Suitable for methods where higher values are better
    },
    "TM_CCOEFF_NORMED": {
        "cv2_method": cv2.TM_CCOEFF_NORMED,
        "threshold": 0.8  # Suitable for normalized methods where higher values indicate better matches
    },
    "TM_CCORR": {
        "cv2_method": cv2.TM_CCORR,
        "threshold": 0.9  # High values are better, threshold can be quite high because it is not normalized
    },
    "TM_CCORR_NORMED": {
        "cv2_method": cv2.TM_CCORR_NORMED,
        "threshold": 0.9  # Typically requires high values for good matches due to normalization
    },
    "TM_SQDIFF": {
        "cv2_method": cv2.TM_SQDIFF,
        "threshold": 0.1  # Lower values are better, inverse relationship
    },
    "TM_SQDIFF_NORMED": {
        "cv2_method": cv2.TM_SQDIFF_NORMED,
        "threshold": 0.1  # Lower values indicate better matches, similar to TM_SQDIFF but normalized
    }
}

def invariantMatchTemplate(rgbimage, rgbtemplate, method_name, rot_range, scale_range):
    if method_name not in methods:
        raise ValueError(f"Unsupported method {method_name}")
    method_info = methods[method_name]

    img_gray = cv2.cvtColor(rgbimage, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(rgbtemplate, cv2.COLOR_RGB2GRAY)

    all_points = []
    for angle in np.arange(rot_range[0], rot_range[1], rot_range[2]):
        for scale in np.arange(scale_range[0], scale_range[1], scale_range[2]):
            scaled_rotated_template = rotate_and_scale(template_gray, angle, scale / 100.0, img_gray.shape)
            points = match_template(img_gray, scaled_rotated_template, method_info)
            for pt in points:
                all_points.append((pt[0], pt[1], angle, scale))

    return all_points

# Continue

def main():
    # Read the BGR image using OpenCV
    img_bgr = cv2.imread("image_1.png")
    if img_bgr is None:
        print("Error loading 'image_1.png'")
        return  # Stop further processing if the image is not read

    # Convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Read the template using matplotlib's imread which reads in RGB format
    template_rgb = plt.imread("template_1.jpg")
    if template_rgb is None or template_rgb.size == 0:
        print("Error loading 'template_1.jpg'")
        return  # Stop further processing if the template is not read

    # Apply cropping to the template
    cropped_template_rgb = template_crop(template_rgb)
    if cropped_template_rgb is None or cropped_template_rgb.size == 0:
        print("Cropping returned an empty image")
        return  # Check if the cropping function failed

    # Display cropped template
    plt.figure(num='Template - Close the Window to Continue >>>')
    plt.imshow(cropped_template_rgb)
    plt.show()

    # Setup parameters for template matching
    method_name = "TM_CCOEFF_NORMED"
    rot_range = [0, 360, 10]  # Start, end, step
    scale_range = [100, 150, 10]  # Start, end, step

    # Perform template matching
    points_list = invariantMatchTemplate(img_rgb, cropped_template_rgb, method_name, rot_range, scale_range)

    # Plot results
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    for point_info in points_list:
        x, y, angle, scale = point_info
        point = (x, y)
        # Calculate rectangle dimensions and rotation
        rect_width, rect_height = calculate_template_dimensions_at_scale(cropped_template_rgb, scale)
        rect = patches.Rectangle(point, rect_width, rect_height, linewidth=1, edgecolor='r', facecolor='none')
        transform = mpl.transforms.Affine2D().rotate_deg_around(point[0] + rect_width / 2, point[1] + rect_height / 2, angle) + ax.transData
        rect.set_transform(transform)
        ax.add_patch(rect)
    plt.show()

if __name__ == "__main__":
    main()
