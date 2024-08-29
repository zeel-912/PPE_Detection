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
