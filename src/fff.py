import os
from functools import partial

import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

import copy
from torch.utils.data import Dataset
import yaml
from PIL import Image
from torchvision import transforms

##########################
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
        for box in boxes:
            labels.append(box['label'])
        
        if self.transform:
            image = self.transform(image)
        
        # Convert labels to a single label (if applicable)
        # If multi-label classification is needed, convert to a one-hot encoding or similar representation
        labels = labels[0]  # Assuming you want to use the first label for now

        return image, labels

# Esempio di utilizzo
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
##########################

dataset = TrafficLightDataset(yaml_file='/home/giuseppe/Scaricati/train/train.yaml', transform=transform)

# Estrai le immagini e le etichette dal dataset
images = []
labels = []
i = 0
for img, label in dataset:
    i += 1
    if i == 1000:
        break
    images.append(img.numpy())  # Converti l'immagine in numpy array
    labels.append(label)

# Verifica se le immagini e le etichette sono state caricate correttamente
print(f"Number of images: {len(images)}")
print(f"Number of labels: {len(labels)}")

# Converti le immagini e le etichette in numpy array
images = np.array(images)
labels = np.array(labels)

# Assicurati che le immagini e le etichette non siano vuote
if len(images) == 0 or len(labels) == 0:
    raise ValueError("Le immagini o le etichette sono vuote. Assicurati che i dati siano caricati correttamente.")

# Suddividi il dataset in training e test set
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Preprocessamento delle immagini
x_train = x_train / 255.0
x_test = x_test / 255.0

# Aggiungi una dimensione per i canali (richiesta dai livelli Conv2D)
x_train = x_train.reshape(x_train.shape[0], 224, 224, 3)  # Assuming RGB images
x_test = x_test.reshape(x_test.shape[0], 224, 224, 3)

NUM_CLIENTS = 5
partition_size = x_train.shape[0] // NUM_CLIENTS
client_id_to_indices = {}
beg_ids = [i * partition_size for i in range(NUM_CLIENTS)]
end_ids = [i * partition_size for i in range(1, NUM_CLIENTS + 1)]
for client_id, (beg_id, end_id) in enumerate(zip(beg_ids, end_ids)):
    client_id_to_indices[client_id] = [beg_id, end_id]

x_split = np.split(x_train, NUM_CLIENTS)
y_split = np.split(y_train, NUM_CLIENTS)
num_data_in_split = x_split[0].shape[0]
train_split = 0.8
x_trains, y_trains, x_tests, y_tests = {}, {}, {}, {}
for idx, (client_x, client_y) in enumerate(zip(x_split, y_split)):
    train_end_idx = int(0.8 * num_data_in_split)
    x_trains[str(idx)] = client_x[:train_end_idx]
    y_trains[str(idx)] = client_y[:train_end_idx]
    x_tests[str(idx)] = client_x[train_end_idx:]
    y_tests[str(idx)] = client_y[train_end_idx:]

class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu')
        self.pool1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.pool2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(10, activation='softmax')  # Assuming 10 classes

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(inputs)
        x = self.pool2(x)
        x = self.flatten()
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.model.build((None, 224, 224, 3))  # Input shape for RGB images
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, batch_size=32, verbose=0)
        return loss, len(self.X_test), {"accuracy": accuracy}

def create_client(cid, model_class, x_trains, y_trains, x_tests, y_tests) -> FlowerClient:
    model = model_class()
    return FlowerClient(model, x_trains[cid], y_trains[cid], x_tests[cid], y_tests[cid])

client_fnc = partial(
    create_client,
    model_class=CNN,
    x_trains=x_trains,
    y_trains=y_trains,
    x_tests=x_tests,
    y_tests=y_tests,
)

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
    min_fit_clients=2,  # Never sample less than 2 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=5,  # Wait until all 5 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fnc,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 0},
    ray_init_args={
        "num_cpus": 1,
        "num_gpus": 0,
        "_system_config": {"automatic_object_spilling_enabled": False},
    },
)
