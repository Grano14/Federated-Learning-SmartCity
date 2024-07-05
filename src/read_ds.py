import os
import yaml
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms


class TrafficLightDataset(Dataset):
    def __init__(self, yaml_file, transform=None):
        with open(yaml_file, 'r') as file:
            self.data = yaml.safe_load(file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['path']
        image = Image.open(img_path).convert('RGB')

        boxes = self.data[idx]['boxes']
        labels = []
        bboxes = []
        for box in boxes:
            labels.append(box['label'])
            bboxes.append([box['x_min'], box['y_min'], box['x_max'], box['y_max']])

        if self.transform:
            image = self.transform(image)

        return image, labels, bboxes


# Esempio di utilizzo
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# dataset = TrafficLightDataset(yaml_file='/home/giuseppe/Scaricati/train/train.yaml', transform=transform)

# Determinare le dimensioni dei dataset di train e test
# dataset_size = len(dataset)
# train_size = int(0.8 * dataset_size)
# test_size = dataset_size - train_size

# Dividere il dataset in train e test
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Verifica
# print("Dimensione del training set:", len(train_dataset))
# print("Dimensione del test set:", len(test_dataset))
