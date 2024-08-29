import os
import xml.etree.ElementTree as ET
import argparse

def convert_bbox(size, box):
    """
    Convert PascalVOC bounding box format to YOLOv8 format.
    Arguments:
        size: tuple of (width, height)
        box: tuple of (xmin, xmax, ymin, ymax)
    Returns:
        Bounding box in YOLOv8 format (x_center, y_center, width, height)
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file, output_dir, classes):
    """
    Convert a single PascalVOC annotation XML file to YOLOv8 format.
    Arguments:
        xml_file: Path to the XML file.
        output_dir: Path to the output directory.
        classes: List of classes.
    """
    print(f"Processing {xml_file}")  # Debug statement
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + '.txt')
    print(f"Output file will be {output_file}")  # Debug statement
    with open(output_file, 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bbox = convert_bbox((width, height), b)
            out_file.write(f"{cls_id} " + " ".join([f"{a:.6f}" for a in bbox]) + '\n')
    print(f"Finished processing {xml_file}")  # Debug statement

def convert_pascal_voc_to_yolo(input_dir, output_person_dir, output_ppe_dir, person_classes, ppe_classes):
    """
    Convert all PascalVOC annotations in a directory to YOLOv8 format for person and PPE separately.
    Arguments:
        input_dir: Path to the input directory containing PascalVOC annotations.
        output_person_dir: Path to the output directory for YOLOv8 person annotations.
        output_ppe_dir: Path to the output directory for YOLOv8 PPE annotations.
        person_classes: List of person classes.
        ppe_classes: List of PPE classes.
    """
    os.makedirs(output_person_dir, exist_ok=True)
    os.makedirs(output_ppe_dir, exist_ok=True)

    # Process all XML files in the input directory
    for xml_file in os.listdir(input_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(input_dir, xml_file)
            # Create separate output directories for person and PPE
            convert_annotation(xml_path, output_person_dir, person_classes)
            convert_annotation(xml_path, output_ppe_dir, ppe_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PascalVOC annotations to YOLOv8 format for person and PPE")
    parser.add_argument("input_dir", help="Path to the input directory containing PascalVOC annotations")
    parser.add_argument("output_person_dir", help="Path to the output directory for YOLOv8 person annotations")
    parser.add_argument("output_ppe_dir", help="Path to the output directory for YOLOv8 PPE annotations")
    parser.add_argument("person_classes_file", help="Path to the file containing the person class names")
    parser.add_argument("ppe_classes_file", help="Path to the file containing the PPE class names")
    args = parser.parse_args()

    # Load person and PPE classes from their respective files
    with open(args.person_classes_file, 'r') as f:
        person_classes = f.read().strip().splitlines()

    with open(args.ppe_classes_file, 'r') as f:
        ppe_classes = f.read().strip().splitlines()

    # Convert annotations
    convert_pascal_voc_to_yolo(args.input_dir, args.output_person_dir, args.output_ppe_dir, person_classes, ppe_classes)
