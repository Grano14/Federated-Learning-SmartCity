import os

# Define the mapping from string class names to numeric class IDs
class_mapping = {
    'Red': 0,
    'Green': 1,
    'Yellow': 2,
    'off': 3,
    'RedLeft' : 0,
    'GreenLeft' : 1,
    'GreenRight' : 1,
    'GreenStraight' : 1,
    'GreenStraightRight' : 1,
    'RedRight' : 0,
    'RedStraight' : 0,
    'RedStraightLeft' : 0,
    'GreenStraightLeft' : 1,
}

# Paths to the label directories
label_dirs = [
    'dataset/train/labels',
    'dataset/val/labels'
]

def convert_labels(label_dir, class_mapping):
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(label_dir, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0 and parts[0] in class_mapping:
                    class_id = class_mapping[parts[0]]
                    new_line = f"{class_id} {' '.join(parts[1:])}"
                    new_lines.append(new_line)
                else:
                    print(f"Warning: Unrecognized label in {filepath}: {line.strip()}")
            with open(filepath, 'w') as file:
                file.write('\n'.join(new_lines) + '\n')

for label_dir in label_dirs:
    if os.path.exists(label_dir):
        convert_labels(label_dir, class_mapping)
    else:
        print(f"Label directory not found: {label_dir}")

print("Labels converted successfully!")
