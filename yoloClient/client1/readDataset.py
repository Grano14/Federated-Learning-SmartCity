import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class CustomMulticlassDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.labels[idx])

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        with open(label_path, 'r') as f:
            line = f.readline().strip().split()
            label = int(line[0])  # Prendi solo la prima cifra come etichetta di classe

        return image, torch.tensor(label)


# Definire le trasformazioni
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Creare il dataset di validazione e il DataLoader
val_dataset = CustomMulticlassDataset(images_dir='dataset/val/images', labels_dir='dataset/val/labels',
                                      transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


# Funzione per valutare il modello
def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            preds = torch.argmax(outputs, dim=1)  # Argmax per classificazione multiclasse

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return all_labels, all_preds


# Funzione per calcolare le metriche
def calculate_metrics(labels, preds):
    print(labels)
    print("\n---------------------------------------\n")
    print(preds)
    accuracy = accuracy_score(labels, preds)

    f1 = f1_score(labels, preds, average='weighted')
    return accuracy, f1

