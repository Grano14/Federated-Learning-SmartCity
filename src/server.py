from flask import Flask, request, jsonify
import random
import copy
import numpy as np    
import tensorflow as tf
from torch import nn
import torch
from torch.utils.data import DataLoader
from utils import get_dataset_bosch
from torch import nn

app = Flask(__name__)

class Client:
    def __init__(self, accuracy, id, flag):
        self.accuracy = accuracy
        self.id = id
        self.flag = flag

    def __str__(self):
        return f'Client(id={self.id}, accuracy={self.accuracy}, flag={self.flag})'

# INIZIALIZZAZIONE DEL MODELLO CENTRALE
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
    global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=1)
    global_model.eval()

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        for images, labels in test_loader:
            # Calcola le predizioni del modello
            outputs = global_model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Calcola le statistiche per la valutazione
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calcola l'accuratezza del modello sui dati di test
    accuracy = correct / total
    print(f'Accuracy on the test set: {accuracy:.4f}')

    return global_model



# Supponiamo che 'model' sia il tuo modello TensorFlow globale
# Addestra o aggiorna il modello con i dati aggregati dai client

central_model = create_model()
# Salva il modello globale
#global_model_path = './src/modello_globale.'
#tf.saved_model.save(central_model, global_model_path)
#print(f"Modello globale salvato in: {global_model_path}")

avg_w = None
client = []
client_flag = [] #SERVE PER TENERE TRACCIA DI QUALI CLIENT DEVONO ESSERE AGGIORNATI


# PARAMETRI PER AFO
update_frequency = 1  # NON SO A QUANTO METTERLA
current_round = 0

#lista dei pesi e dell'accuracy ricevuti dai client
weights = []
accuracy = []

#funzione per il calcolo della media dei pesi
"""def average_weights(w, accuracy):

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
"""

def average_weights(w):
    w_avg = []
    
    for item in w:
        w_avg.append(copy.deepcopy(item))

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def client_to_update(accuracy):
    
    client_updated = 0
    a_avg = accuracy.mean()

    sorted_client = sorted(client, key=lambda x: x.accuracy, reverse=False)

    for item in sorted_client:
        if item.accuracy < a_avg:
            item.flag = True
            client_updated = client_updated + 1
        if client_updated < 2:
            item.flag = True
            client_updated = 1 + client_updated

def find_client_by_id(clients, client_id):
    for item in clients:
        if item.id == client_id:
            return item
    return None 

@app.route('/get_new_weight', methods=['GET'])
def get_new_weight():    

    new_client = find_client_by_id(request.json.get('id'))

    if new_client != None and new_client.flag:
        new_client.flag == False
        return 'Accesso consentito', avg_w, 200

    return 'Accesso negato', 400, 0




@app.route('/upload_client_weights', methods=['POST'])
def upload_model_weights():

    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        file.save('./src/received_model_weights.pth')
        model_weights = torch.load('./src/received_model_weights.pth')
        weights.append(model_weights)
    else: 
        return 'Pesi non memorizzati', 400  

    accuracy.append(request.json.get('accuracy'))
    if len(weights) == 6: 
        avg_w = average_weights(weights)
        client_to_update(accuracy)
        weights.clear()
        accuracy.clear()

    return 'Pesi inviati correttamente', 200
  





@app.route('/connect', methods=['GET'])   #TENGO TRACCIA DEI CLIENT CONNESSI
def connect_client():

    
    new_client = Client(None, request.json.get('id'), True)

    for item in client:
        if item.id == new_client.id:
            return 'client non connesso - chiave già inserita', 400 
    client.append(new_client)

    return jsonify({"status": "Client connected", "model": loaded_model}), 200


@app.route('/send_model', methods=['POST'])     #FUNZIONE PER INVIO MODELLO CENTRALE AI CLIENT
def send_model():
    global connected_clients
    
    # Copia il modello centrale attuale
    current_model = copy.deepcopy(central_model)
    
    # Invia il modello a un numero casuale di client (nel tuo caso, 6)
    selected_clients = random.sample(connected_clients, 6)
    
    for client in selected_clients:
        # Simula l'invio del modello al client (sostituire con implementazione reale)
        client_url = f"http://{client['ip']}:{client['port']}/receive_model"
        response = requests.post(client_url, json={"model_weights": current_model.get_weights()})
        if response.status_code == 200:
            print(f"Model sent to {client['ip']}:{client['port']}")
        else:
            print(f"Failed to send model to {client['ip']}:{client['port']}, status code {response.status_code}")

    return jsonify({"status": "Model sent to selected clients"}), 200







@app.route('/upload_gradients', methods=['POST'])
def upload_gradients():
    global client_gradients, current_round
    data = request.json
    print("Data received:")  # Log the received data
    gradients = np.array(data['gradients'])
    client_gradients.append(gradients)
    
    # Check if we have gradients from all clients
    if len(client_gradients) == 6:
        # (FedAvg)
        average_gradients = np.mean(client_gradients, axis=0)
        
        # Applicazione dei pesi calcolati al modello centrale
        apply_gradients(average_gradients)
        
        client_gradients = []
        
        current_round += 1
    
    return jsonify({"status": "Gradients received"}), 200

def apply_gradients(average_gradients):
    optimizer = central_model.optimizer
    with tf.GradientTape() as tape:
        central_model.train_on_batch(x=np.zeros((1, 64, 64, 3)), y=np.zeros((1, 1)))
    weights = central_model.trainable_weights
    optimizer.apply_gradients(zip(average_gradients, weights))

@app.route('/get_model', methods=['GET'])
def get_model():
    weights = central_model.get_weights()
    return jsonify({"weights": [w.tolist() for w in weights]}), 200

@app.route('/update_frequency', methods=['POST'])
def update_frequency():
    global update_frequency
    data = request.json
    update_frequency = data['update_frequency']
    return jsonify({"status": "Update frequency set"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
