from flask import Flask, request, jsonify
import random
import copy
import numpy as np    
import tensorflow as tf
import torch

from utils import get_dataset_bosch
from torch import nn

app = Flask(__name__)


# INIZIALIZZAZIONE DEL MODELLO CENTRALE
def create_model():

    class MLP(tf.Module):
        def __init__(self, dim_in, dim_hidden, dim_out):
            super(MLP, self).__init__()
            self.layer_input = tf.keras.layers.Dense(dim_hidden, activation='relu', input_dim=dim_in)
            self.dropout = tf.keras.layers.Dropout(0.5)
            self.layer_hidden = tf.keras.layers.Dense(dim_out, activation='sigmoid')

        def __call__(self, inputs):
            x = self.layer_input(inputs)
            x = self.dropout(x)
            return self.layer_hidden(x)
        

    train_dataset, test_dataset = get_dataset_bosch()

    img_size = train_dataset[0][0].shape
    len_in = 1
    for x in img_size:
        len_in *= x
    global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=1)
    return global_model



# Supponiamo che 'model' sia il tuo modello TensorFlow globale
# Addestra o aggiorna il modello con i dati aggregati dai client

central_model = create_model()
# Salva il modello globale
global_model_path = './src/modello_globale.'
tf.saved_model.save(central_model, global_model_path)
print(f"Modello globale salvato in: {global_model_path}")


client_gradients = []


# PARAMETRI PER AFO
update_frequency = 1  # NON SO A QUANTO METTERLA
current_round = 0

weights = []

#funzione per il calcolo della media dei pesi
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

####################################
@app.route('/upload_model_weights', methods=['POST'])
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
        print(len(weights))
        if len(weights) == 5:
            avg_w = average_weights(weights)
            print(avg_w)
        return 'File successfully uploaded', 200

####################################



@app.route('/connect', methods=['GET'])   #TENGO TRACCIA DEI CLIENT CONNESSI
def connect_client():
    #global connected_clients
    #client_info = request.json
    #connected_clients.append(client_info)
    #print(f"Client connected: {client_info['ip']}:{client_info['port']}")

    global_model_path = './src/modello_globale.'

    #Carica il modello globale
    loaded_model = tf.saved_model.load(global_model_path)
    #print("Modello globale caricato con successo.")

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
