import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import get_dataset_bosch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import requests
import io


# CREAZIONE MODELLO (UNIFORME SU TUTTI I CLIENT + SERVER)
def create_model():
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

    train_dataset, test_dataset = get_dataset_bosch()

    img_size = train_dataset[0][0].shape
    len_in = 1
    for x in img_size:
        len_in *= x
    global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=5)  # Assuming 4 classes
    return global_model, train_dataset, test_dataset

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

def main():
    # Creazione modello
    model, train_dataset, test_dataset = create_model()
    """"
    batch_size = 32
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Addestriamo il modello
    train_model(model, train_loader, criterion, optimizer, num_epochs=1)
    
    validate_model(model, test_loader)

    # Salvataggio dei pesi del modello
    torch.save(model.state_dict(), './src/model_weights.pth')
    print("Model weights saved to 'model_weights.pth'")
    """
    model.load_state_dict(torch.load('./src/model_weights.pth'))
    print(model.state_dict())

    model_weights = model.state_dict()

    # Serializza i pesi del modello in un buffer binario
    buffer = io.BytesIO()
    torch.save(model_weights, buffer)
    buffer.seek(0)

    # URL del server a cui inviare i pesi del modello
    server_url = 'http://localhost:5000/upload_model_weights'

    # Invia una richiesta POST con i pesi del modello
    response = requests.post(server_url, files={'file': buffer})

    # Controlla la risposta del server
    if response.status_code == 200:
        print('Model weights successfully uploaded')
    else:
        print('Failed to upload model weights:', response.content)


if __name__ == "__main__":
    main()





