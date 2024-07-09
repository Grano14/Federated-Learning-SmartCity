from flask import Flask, request, jsonify
import random
import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

app = Flask(__name__)

central_model = None
client_gradients = []

# PARAMETRI PER AFO
update_frequency = 1  # NON SO A QUANTO METTERLA
current_round = 0


@app.route('/connect', methods=['POST'])   #TENGO TRACCIA DEI CLIENT CONNESSI
def connect_client():
    global connected_clients
    client_info = request.json
    connected_clients.append(client_info)
    print(f"Client connected: {client_info['ip']}:{client_info['port']}")
    return jsonify({"status": "Client connected"}), 200


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




# INIZIALIZZAZIONE DEL MODELLO CENTRALE
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=SGD(), loss='binary_crossentropy', metrics=['accuracy'])
    central_model = model



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
