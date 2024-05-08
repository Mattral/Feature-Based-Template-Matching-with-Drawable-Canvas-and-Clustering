# Feature-Based Template Matching with Drawable Canvas and Clustering

This repository contains a Streamlit application that demonstrates feature-based template matching using the SIFT (Scale-Invariant Feature Transform) algorithm, enhanced with DBSCAN clustering for robust localization. Users can upload an image and a template, then manually select the area of interest within the template using a drawable canvas. The application uses the adjusted Lowe ratio to match features and DBSCAN clustering to identify and localize the matched region in the main image.

latest ver -
old ver -

## Features

- Upload and display images directly in the browser.
- Interactive canvas to define the region of interest in the template image.
- Adjustable Lowe ratio for fine-tuning feature matching sensitivity.
- Adjustable `eps` and `min_samples` for DBSCAN clustering to refine how clusters are formed.
- Visual results showing matched points and the bounding boxes of matched areas, aiding in precise template localization.

## Installation

To run this project locally, you need Python and Streamlit. Follow these steps:

1. Clone this repository:
   ```bash
   git clone [This repo link.git]
   cd feature-based-template-matching-with-drawable-canvas
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage

After starting the application, follow these steps:

1. **Upload Your Images**: Use the sidebar to upload your main image and the template image.
2. **Draw the Template Area**: On the displayed template, use the mouse to draw a rectangle around the area you want to match.
3. **Adjust the Lowe Ratio and DBSCAN Parameters**: Use the sidebar sliders to adjust the Lowe ratio and DBSCAN parameters (`eps` and `min_samples`) for clustering.
4. **Match Template**: Click the "Match Template" button to perform the feature matching and clustering. View the results in the main panel, which includes match points and clustered bounding boxes.

## Technologies Used

- **Streamlit**: An open-source app framework for Machine Learning and Data Science projects.
- **OpenCV**: A library of programming functions mainly aimed at real-time computer vision.
- **NumPy**: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.
- **Pillow (PIL Fork)**: The Python Imaging Library adds image processing capabilities to your Python interpreter.
- **streamlit-drawable-canvas**: A Streamlit component for creating interactive canvas.
- **Scikit-Learn**: For implementing the DBSCAN clustering algorithm.

## Acknowledgments

This project utilizes several open-source packages that make modern computer vision techniques and data clustering accessible and easy to implement. I want to thank the developers and maintainers of these packages for their valuable contributions to the open-source community.

## License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.

---
Feel free to star this repository if you find it useful! For help and support, create an issue, and I'll get back to you.
