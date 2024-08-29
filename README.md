PPE Detection and Person Detection using YOLOv8
Overview
This project involves training and deploying YOLOv8 models for detecting persons and Personal Protective Equipment (PPE) items. The process includes converting annotations, training models, and performing inference on images.

Steps
Annotation Conversion:

Converted PascalVOC annotations to YOLOv8 format for both person detection and PPE detection classes. This step ensures compatibility with YOLOv8's input format.
Model Training:

Utilized Roboflow to create datasets and train YOLOv8 models for:
Person Detection: Identifying persons in images.
PPE Detection: Identifying various PPE items such as hard hats, gloves, masks, glasses, boots, vests, PPE suits, ear protectors, and safety harnesses.
Inference and Image Processing:

Applied the trained person detection model to input images to generate cropped images around detected persons.
Processed these cropped images with the trained PPE detection model to identify PPE items.
Saved the results of PPE detection to a directory named ppe_detection_images.
Requirements
Python 3.7+
OpenCV
Roboflow Python library
YOLOv8
Installation
Install the required Python libraries:

bash
Copy code
pip install opencv-python roboflow
Usage
Convert Annotations:

Run the pascalVOC_to_yolo.py script to convert annotations from PascalVOC to YOLOv8 format.
Train Models:

Train YOLOv8 models using Roboflow for both person detection and PPE detection.
Run Inference:

Execute the inference script to process images through the person detection and PPE detection models.
Ensure that the input_dir contains the images for inference and that results are saved in the ppe_detection_images directory.
Scripts
pascalVOC_to_yolo.py: Converts PascalVOC annotations to YOLOv8 format.
inference.py: Runs inference using the trained YOLOv8 models for person and PPE detection.
Notes
Ensure to replace the API key in the scripts with your own Roboflow API key.
The output images with detected PPE items will be saved in the ppe_detection_images directory.
Contact
For any questions or issues, please contact Zeel Patel.
