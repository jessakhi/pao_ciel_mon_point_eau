
# Ciel Mon Point d’Eau: Hydraulic Object Detection from Satellite Data

## Project Overview

This project, "Ciel Mon Point d’Eau," is focused on the extraction of hydraulic object boundaries from satellite data using deep learning techniques. The project leverages data from satellites such as SWOT (Surface Water and Ocean Topography), Sentinel S1, and Sentinel S2, with the aim of identifying water bodies in Brazil's semi-arid Ceará region. This work was conducted in collaboration with the University of Ceará in Brazil, with support from CNES, NASA, and guidance from Dr. Marielle Gosset and Mr. Gilles Gasso.

## Features

- **Data Selection**: Carefully selected satellite images (30 GB reduced to a 4.5 GB dataset) were sourced from the SWOT, Sentinel S1, and Sentinel S2 satellites. Images were chosen based on minimal interference and maximum data quality, processed using QGIS for geospatial analysis.

- **Image Preprocessing**:
  - **Imagette Creation**: Original large-scale images were divided into smaller 256x256-pixel "imagettes" for efficient deep learning processing.
  - **Selection Threshold**: Imagettes were chosen to maintain a balanced representation of water and land (at least 10% each) to ensure diverse data for model training.
  - **Normalization**: Imagette pixel values were scaled between 0 and 1 to improve model performance.

- **Deep Learning Model**:
  - **U-Net Architecture**: U-Net, a popular model for image segmentation, was employed for this project to ensure precise pixel-level classification.
  - **Model Configuration**: Configured with a dropout of 0.5 and batch normalization, optimized using the Adam optimizer.
  - **Loss Function**: Combined Binary Cross-Entropy and Dice Loss to address class imbalances between water and land areas.

## Dataset Preparation

The final dataset includes 2,096 imagettes, offering a diverse representation of the area's topography. The imagettes were further processed using rasterio in Python, ensuring compatibility with deep learning models and enhancing spatial continuity by allowing overlapping areas between adjacent imagettes.

## Model Training and Testing

1. **Initial Experiments**: 
   - Early trials utilized a simple U-Net with only two convolution layers due to resource limitations, providing a foundation for further development.
   - Subsequent adjustments included variations in epochs and batch sizes to optimize learning without overfitting.

2. **Post-Processing**:
   - Predicted images were binarized to create binary masks identifying water pixels using OpenCV’s adaptive binarization (Otsu’s thresholding).

3. **Results and Conclusions**:
   - The model now effectively predicts water body masks under certain conditions. However, stability improvements are ongoing, and further parameter tuning is planned to achieve consistent loss reduction.

## Installation and Setup

1. Clone the repository.
2. Set up the environment with required packages:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the model:
   ```sh
   python run_model.py
   ```

## Usage

1. **Data Preprocessing**: Execute data selection and preprocessing scripts to prepare satellite images.
2. **Model Training**: Adjust configuration parameters as needed and train the U-Net model.
3. **Post-Processing**: Use the provided binarization script for mask output.

## Contributors

- Rei Ito
- Jihane Essakhi
- Pierre-Marie Stevenin

## Acknowledgements

This project was made possible through the support of Dr. Marielle Gosset, CNES, NASA, and Mr. Gilles Gasso for project guidance.

## License

This project is open-source under the MIT License.

--- 
