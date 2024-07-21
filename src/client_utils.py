import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import get_dataset_bosch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import requests
import io
import time


# CREAZIONE MODELLO E DEL DATASET
def create_model_and_dataset(dataset_str):
    class MLP(nn.Module):
        def __init__(self, dim_in, dim_hidden, dim_out):
            super(MLP, self).__init__()
            self.layer_input = nn.Linear(dim_in, dim_hidden)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.5)
            self.layer_hidden = nn.Linear(dim_hidden, dim_out)

        def forward(self, x):
            x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])  # Flatten the input
            x = self.layer_input(x)  # Apply input layer
            x = self.dropout(x)  # Apply dropout
            x = self.relu(x)  # Apply ReLU activation
            x = self.layer_hidden(x)  # Apply hidden layer
            return x  # Output logits directly for CrossEntropyLoss

    #ottenimento del dataset
    train_dataset, test_dataset = get_dataset_bosch(dataset_str)

    img_size = train_dataset[0][0].shape
    len_in = 1
    for x in img_size:
        len_in *= x
    #creazionde le modello
    global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=5)  # Assuming 4 classes
    return global_model, train_dataset, test_dataset


#funzione per allenare il modello sui dati
def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


#funzione per valutare le prestazioni del modello
def validate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test set: {100 * correct / total}%')
    return {100 * correct / total}

#funzione per l'invio dei pesi al server
def send_weights(model_weights, accuracy, id):
    # Serializza i pesi del modello in un buffer binario
    buffer = io.BytesIO()
    torch.save(model_weights, buffer)
    buffer.seek(0)

    # URL del server a cui inviare i pesi
    server_url_upload = 'http://172.20.10.9:5000/upload_client_weights'

    # Invia una richiesta POST con i pesi del modello e l'accuracy
    data = {'accuracy': accuracy, "id": id}  
    response = requests.post(server_url_upload, files={'file': buffer}, data=data)
    print(response.text)
    # Controlla la risposta del server
    if response.status_code == 200:
        print('Model weights successfully uploaded')
    else:
        print('Failed to upload model weights:', response.content)


