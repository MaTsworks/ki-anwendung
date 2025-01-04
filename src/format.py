import os
import xml.etree.ElementTree as ET
import argparse

def parse_voc_annotation(xml_file, labels):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/height").text)
    yolo_annotations = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in labels:
            continue

        class_id = labels.index(class_name)
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Normalize coordinates
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        # YOLO format: <class_id> <x_center> <y_center> <width> <height>
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_annotations

def convert_annotations(voc_dir, output_dir, labels_file):
    with open(labels_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)

    for xml_file in os.listdir(voc_dir):
        if not xml_file.endswith(".xml"):
            continue

        input_path = os.path.join(voc_dir, xml_file)
        base_name = os.path.splitext(xml_file)[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        yolo_annotations = parse_voc_annotation(input_path, labels)

        with open(output_path, "w") as f:
            f.write("\n".join(yolo_annotations))
        print(f"Converted {xml_file} to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pascal VOC annotations to YOLO format")
    parser.add_argument("--voc_dir", type=str, required=True, help="Path to directory with Pascal VOC XML files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save YOLO annotations")
    parser.add_argument("--labels_file", type=str, required=True, help="Path to file containing class labels")
    args = parser.parse_args()

    convert_annotations(args.voc_dir, args.output_dir, args.labels_file)
