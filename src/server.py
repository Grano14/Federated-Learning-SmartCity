from flask import Flask, request, jsonify, send_file
import random
import copy
import numpy as np    
import tensorflow as tf
from torch import nn
import torch
from torch.utils.data import DataLoader
from utils import get_dataset_bosch
from torch import nn
import io

app = Flask(__name__)

# STRUTTURA CLIENT
class Client:
    def __init__(self, accuracy, id, flag, attendi):
        self.accuracy = accuracy
        self.id = id
        self.flag = flag
        self.attendi = attendi

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

    return global_model


central_model = create_model()
# Salva il modello globale
#global_model_path = './src/modello_globale.'
#tf.saved_model.save(central_model, global_model_path)
#print(f"Modello globale salvato in: {global_model_path}")

client = [] #SERVE PER TENERE TRACCIA DEI CLIENT

#lista dei pesi e dell'accuracy ricevuti dai client
weights = []
accuracy = []

#funzione per il calcolo della media dei pesi
def average_weights(w):
    
    global prova 

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    prova = w_avg
    return w_avg

# FUNZIONE PER IL CALCOLO DELLA MEDIA
def mean(lista):
    if len(lista) == 0:
        return None 
    
    somma = sum(lista)
    media = somma / len(lista)
    return media

#SCELTA DEI CLIENT DA AGGIORNARE
def client_to_update(accuracy):
    
    client_updated = 0
    a_avg = mean(accuracy)
    

    sorted_client = sorted(client, key=lambda x: x.accuracy, reverse=False)
    print('FUNZINE  CLIENT_TO_UPDATE -----')
    for item in sorted_client:
        if item.accuracy < a_avg or client_updated < 1:
            item.flag = True
            client_updated = client_updated + 1
            print("id client da aggiornare------>",  item.id)
        item.attendi = False
        print('Client = ', item.id, '  ----  attendi = ', item.attendi, '   ---   flag = ', item.flag)
    print('FINE     FUNZINE  CLIENT_TO_UPDATE -----')
    
# RICERCA DEI CLIENT PER ID
def find_client_by_id(client_id):
    for item in client:
        if item.id == int(client_id):
            return item
    return None 

# INVIO DEI CODICI E DEI PERMESSI AI CLIENT IN BASE ALLA SCELTA EFFETTUATA PRIMA SU QUALI CLIENT DA AGGIORNARE
@app.route('/get_permission', methods=['GET'])
def get_new_weight():    
    print('FUNZIONE GET_PERMISSION --------')
    n_client = find_client_by_id(request.json.get('id'))
    print('Client = ', n_client.id, '  ----  attendi = ', n_client.attendi, '   ---   flag = ', n_client.flag)

    if n_client != None and n_client.flag and (not n_client.attendi):
        
        data = {
            'codice': 1,
        }
        n_client.flag == False
        n_client.attendi = True
        return jsonify(data)
    else:
        if n_client.attendi:
            data = {
                'codice': 2,
            }
            return jsonify(data)
        else:
            data = {
                'codice': 0,
            }
            n_client.flag == False
            n_client.attendi = True
            return jsonify(data)


 #####GET DEI NUOVI PESI VERSO IL CLIENT
@app.route('/get_model_weight', methods=['GET'])
def get_model_weights():
    buffer = io.BytesIO()
    torch.save(prova, buffer)
    buffer.seek(0)  # Riporta il puntatore all'inizio del buffer

    # Invia il buffer come allegato binario
    return send_file(buffer, mimetype='application/octet-stream')


#### CARICAMENTO DEI NUOVI PESI DA PARTE DEI CLIENT
@app.route('/upload_client_weights', methods=['POST'])
def upload_model_weights():

    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file.save('./src/received_model_weights.pth')
        model_weights = torch.load('./src/received_model_weights.pth')
        weights.append(model_weights)
    else: 
        return 'Pesi non memorizzati' 
    
    new_client = find_client_by_id(request.form.get('id'))
    if new_client != None:
        new_client.accuracy = int(request.form.get('accuracy'))
        accuracy.append(int(request.form.get('accuracy')))
    else:
        return 'Client non trovato'
    
    print(len(weights))
    if len(weights) == 3: 
        average_weights(weights)
        client_to_update(accuracy)
        weights.clear()

    return 'Pesi inviati correttamente'
  


@app.route('/connect', methods=['GET'])   #TENGO TRACCIA DEI CLIENT CONNESSI
def connect_client():

    new_client = Client(None, request.json.get('id'), False, True)
    print('lunghezza client nella connect------>',  len(client))
    for item in client:
        if item.id == new_client.id:
            return 'client non connesso - chiave già inserita'
    client.append(new_client)
    print(client)
    return 'client inserito'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



"""

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

    return jsonify({"status": "Model sent to selected clients"})







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
"""

