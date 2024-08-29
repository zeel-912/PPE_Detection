from roboflow import Roboflow
from PIL import Image, ImageDraw
import os
import cv2
import numpy as np

# Initialize Roboflow and load the PPE detection model
rf = Roboflow(api_key="pOFqoV5xaAQvBkPTALfh")

# Load PPE Detection Model
ppe_project = rf.workspace("syook-assessment").project("ppe-detection-onaod")
ppe_model = ppe_project.version("1").model

# Set the path to the cropped images and output directory for PPE detection results
cropped_dir = "images"
output_dir = "ppe_detected_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over all cropped images
for cropped_image_name in os.listdir(cropped_dir):
    cropped_img_path = os.path.join(cropped_dir, cropped_image_name)

    # Perform PPE detection on the cropped image
    ppe_results = ppe_model.predict(cropped_img_path).json()

    # Convert cropped image to PIL for drawing
    cropped_pil = Image.open(cropped_img_path)
    draw = ImageDraw.Draw(cropped_pil)

    # Draw bounding boxes for detected PPE on the cropped image
    for ppe_bbox in ppe_results['predictions']:
        ppe_x_center = ppe_bbox['x']
        ppe_y_center = ppe_bbox['y']
        ppe_width = ppe_bbox['width']
        ppe_height = ppe_bbox['height']

        top_left = (ppe_x_center - ppe_width / 2, ppe_y_center - ppe_height / 2)
        bottom_right = (ppe_x_center + ppe_width / 2, ppe_y_center + ppe_height / 2)

        draw.rectangle([top_left, bottom_right], outline="red", width=3)

        # Draw the confidence and class label
        label = f"{ppe_bbox['class']} ({ppe_bbox['confidence']:.2f})"
        draw.text((top_left[0], top_left[1] - 10), label, fill="red")

    # Save the final image with PPE detection results
    final_output_path = os.path.join(output_dir, f"ppe_detected_{cropped_image_name}")
    cropped_pil.save(final_output_path)
    print(f"Saved PPE detection result to {final_output_path}")

cv2.destroyAllWindows()
