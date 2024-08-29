import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from roboflow import Roboflow

# Initialize Roboflow and load the models
rf = Roboflow(api_key="pOFqoV5xaAQvBkPTALfh")

# Load Person Detection Model
per_project = rf.workspace("syook-assessment").project("person-detection-oijmo")
person_model = per_project.version("1").model

# Load PPE Detection Model
ppe_project = rf.workspace("syook-assessment").project("ppe-detection-onaod")
ppe_model = ppe_project.version("1").model

def perform_person_detection(input_image_path, temp_dir):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Perform person detection
    person_results = person_model.predict(input_image_path).json()
    image = cv2.imread(input_image_path)

    cropped_images_paths = []

    # Process each detected person
    for i, bbox in enumerate(person_results['predictions']):
        x_center, y_center, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Crop the image around the detected person
        cropped_img = image[y_min:y_max, x_min:x_max]
        cropped_img_path = os.path.join(temp_dir, f"cropped_{i}.jpg")
        cv2.imwrite(cropped_img_path, cropped_img)
        cropped_images_paths.append(cropped_img_path)

    return cropped_images_paths

def perform_ppe_detection(cropped_images_paths, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, cropped_img_path in enumerate(cropped_images_paths):
        # Convert cropped image to PIL format for PPE detection
        cropped_pil = Image.open(cropped_img_path)
        ppe_results = ppe_model.predict(cropped_img_path).json()

        # Draw bounding boxes for detected PPE items
        draw = ImageDraw.Draw(cropped_pil)
        for ppe_bbox in ppe_results['predictions']:
            ppe_x_center = ppe_bbox['x']
            ppe_y_center = ppe_bbox['y']
            ppe_width = ppe_bbox['width']
            ppe_height = ppe_bbox['height']

            top_left = (ppe_x_center - ppe_width / 2, ppe_y_center - ppe_height / 2)
            bottom_right = (ppe_x_center + ppe_width / 2, ppe_y_center + ppe_height / 2)
            
            draw.rectangle([top_left, bottom_right], outline="red", width=3)
            label = f"{ppe_bbox['class']} ({ppe_bbox['confidence']:.2f})"
            draw.text((top_left[0], top_left[1] - 10), label, fill="red")

        # Save the output image with PPE detection results
        output_path = os.path.join(output_dir, f"ppe_detected_{i}.jpg")
        cropped_pil.save(output_path)
        print(f"Saved PPE detection result to {output_path}")

if __name__ == "__main__":
    input_image_path = "final/datasets/images/-184-_png_jpg.rf.b02963998a79b9ad5079f57b65130bc2.jpg"  # Replace with your input image path
    temp_dir = "temp_cropped_images"
    output_dir = "output"

    # Perform person detection and cropping
    cropped_images_paths = perform_person_detection(input_image_path, temp_dir)

    # Perform PPE detection on cropped images
    perform_ppe_detection(cropped_images_paths, output_dir)
