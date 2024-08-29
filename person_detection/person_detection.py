import os
import cv2
from roboflow import Roboflow

# Initialize Roboflow and load the person detection model
rf = Roboflow(api_key="pOFqoV5xaAQvBkPTALfh")

# Load Person Detection Model
per_project = rf.workspace().project("person-detection-oijmo")
person_model = per_project.version("1").model

# Set the path to the test images and output directory for cropped images
input_dir = "images/"
cropped_dir = "cropped_images/"
if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)

# Iterate over all images in the input directory
for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    image = cv2.imread(image_path)

    # Perform person detection
    person_results = person_model.predict(image_path).json()

    for i, bbox in enumerate(person_results['predictions']):
        x_center, y_center, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Add a margin to the cropping (optional)
        margin = 10
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(image.shape[1], x_max + margin)
        y_max = min(image.shape[0], y_max + margin)

        # Crop the image around the detected person
        cropped_img = image[y_min:y_max, x_min:x_max]

        # Save the cropped image
        cropped_img_path = os.path.join(cropped_dir, f"cropped_{i}_{image_name}")
        cv2.imwrite(cropped_img_path, cropped_img)
        print(f"Saved cropped image to {cropped_img_path}")
