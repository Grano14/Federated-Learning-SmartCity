import yaml
import os
from PIL import Image


def convert_to_yolo_format(image_path, bboxes, output_dir):
    img = Image.open(image_path)
    img_width, img_height = img.size

    yolo_annotations = []
    for bbox in bboxes:
        label = bbox['label']
        x_min = bbox['x_min']
        y_min = bbox['y_min']
        x_max = bbox['x_max']
        y_max = bbox['y_max']

        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        yolo_annotation = f"{label} {x_center} {y_center} {width} {height}"
        yolo_annotations.append(yolo_annotation)

    # Generate label file path
    label_filename = os.path.basename(image_path).replace('.png', '.txt')
    label_path = os.path.join(output_dir, label_filename)

    # Save YOLO annotations to label file
    with open(label_path, 'w') as label_file:
        label_file.write("\n".join(yolo_annotations))

    print(f"Saved annotations for {image_path} to {label_path}")


def process_yaml(yaml_file, img_dir, label_dir):
    with open(yaml_file, 'r') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file {yaml_file}: {exc}")
            return

    if not isinstance(data, list):
        print(f"Error: {yaml_file} does not contain a list of annotations")
        return

    for item in data:
        img_path = os.path.join( item['path'])

        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            continue

        bboxes = item.get('boxes', [])

        if not bboxes:
            print(f"Warning: No boxes found for image {img_path}")
            continue

        convert_to_yolo_format(img_path, bboxes, label_dir)


# Paths to your files
train_yaml = 'dataset/train/train.yaml'
test_yaml = 'dataset/test/test.yaml'

# Directories for images and labels
train_img_dir = 'dataset/train/images'
test_img_dir = 'dataset/test/images'
train_label_dir = 'dataset/train/labels'
test_label_dir = 'dataset/test/labels'

# Ensure the label directories exist
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# Process the YAML files
print("Processing train YAML file...")
process_yaml(train_yaml, train_img_dir, train_label_dir)

print("Processing test YAML file...")
process_yaml(test_yaml, test_img_dir, test_label_dir)
