# PPE Detection and Person Detection using YOLOv8

## Overview
This project involves training and deploying YOLOv8 models for detecting persons and Personal Protective Equipment (PPE) items. The process includes converting annotations, training models, and performing inference on images.

## Steps

### Annotation Conversion
- Converted PascalVOC annotations to YOLOv8 format for both person detection and PPE detection classes. This step ensures compatibility with YOLOv8's input format.

### Model Training
- Utilized Roboflow to create datasets and train YOLOv8 models for:
  - **Person Detection**: Identifying persons in images.
  - **PPE Detection**: Identifying various PPE items such as hard hats, gloves, masks, glasses, boots, vests, PPE suits, ear protectors, and safety harnesses.

### Inference and Image Processing
- Applied the trained person detection model to input images to generate cropped images around detected persons.
- Processed these cropped images with the trained PPE detection model to identify PPE items.
- Saved the results of PPE detection to a directory named `ppe_detection_images`.

## Requirements
- Python 3.7+
- OpenCV
- Roboflow Python library
- YOLOv8

## Installation
Install the required Python libraries:

```bash
pip install opencv-python roboflow
```

## Usage

### Convert Annotations
1. Run the `pascalVOC_to_yolo.py` script to convert annotations from PascalVOC to YOLOv8 format.

   ```bash
   python pascalVOC_to_yolo.py --input_dir <input_directory> --output_dir <output_directory>
   ```

   Replace `<input_directory>` with the path to your PascalVOC annotations and `<output_directory>` with the path where YOLOv8 formatted annotations should be saved.

### Train Models
1. Train YOLOv8 models using Roboflow:
   - Log in to Roboflow and create a new project for person detection and another for PPE detection.
   - Upload the converted YOLOv8 dataset to Roboflow.
   - Follow Roboflow's instructions to train the models.

### Run Inference
1. Execute the `inference.py` script to process images through the person detection and PPE detection models.

   ```bash
   python inference.py --input_dir <input_directory> --output_dir <output_directory>
   ```

   - `<input_directory>`: Directory containing images for inference.
   - `<output_directory>`: Directory where results, including images with detected PPE items, will be saved.

## Scripts
- `pascalVOC_to_yolo.py`: Converts PascalVOC annotations to YOLOv8 format.
- `inference.py`: Runs inference using the trained YOLOv8 models for person and PPE detection.

## Notes
- Ensure to replace the API key in the scripts with your own Roboflow API key.
- The output images with detected PPE items will be saved in the `ppe_detection_images` directory.

## Contact
For any questions or issues, please contact patelzeelpramodbhai@gmail.com.
