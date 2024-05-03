import streamlit as st
from Modified import template_crop, invariantMatchTemplate  # Import necessary functions

def main():
    st.title('Image Template Matching')

    # Image Upload
    img_file = st.file_uploader("Upload your main image", type=['png', 'jpg', 'jpeg'])
    template_file = st.file_uploader("Upload your template image", type=['png', 'jpg', 'jpeg'])

    if img_file and template_file:
        # Convert the uploaded file to an OpenCV image
        import cv2
        from PIL import Image
        import numpy as np

        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        template_bytes = np.asarray(bytearray(template_file.read()), dtype=np.uint8)
        template = cv2.imdecode(template_bytes, 1)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

        if st.button('Crop Template'):
            cropped_template = template_crop(template)  # assuming this function handles the UI internally
            if cropped_template is not None:
                st.image(cropped_template, caption='Cropped Template', use_column_width=True)
        
        method_name = st.selectbox('Choose Matching Method', options=['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR', 'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED'])
        if st.button('Match Template'):
            rot_range = [0, 360, 10]  # Define as needed
            scale_range = [100, 150, 10]  # Define as needed
            points_list = invariantMatchTemplate(img, cropped_template, method_name, rot_range, scale_range)
            # Visualization code here using st.pyplot or other compatible methods

if __name__ == '__main__':
    main()
