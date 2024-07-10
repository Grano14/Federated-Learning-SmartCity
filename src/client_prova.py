import numpy as np
import tensorflow as tf
import requests
import json
from flask import Flask, request, jsonify

from utils import get_dataset_bosch
from sklearn.model_selection import train_test_split

app = Flask(__name__)

client_ip = '192.168.1.10'
client_port = 8000

CONNECT_URL = 'http://127.0.0.1:5000/connect'
UPLOAD_URL = 'http://127.0.0.1:5000/upload_gradients'
MODEL_URL = 'http://127.0.0.1:5000/get_model'

# CREAZIONE MODELLO (UNIFORME SU TUTTI I CLIENT + SERVER)
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



@app.route('/receive_model', methods=['POST'])
def receive_model():
    global central_model
    model_weights = request.json['model_weights']
    model.set_weights(model_weights)
    return jsonify({"status": "Model weights received"}), 200

def download_updated_model():                   #PRENDO IL MODELLO AGGIORNATO DAL BOSS DEL POPPIN
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        model_weights = json.loads(response.text)['weights']
        return model_weights
    else:
        raise RuntimeError(f"Failed to download model weights: {response.status_code}")
    
def integrate_updated_model(model, weights):   #SERVE AD AGGIORNARE IL MODELLO SUL CLIENT
    model.set_weights(weights)


def get_gradients(model, x, y):              #CALCOLO I PESI DEL MODELLO
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = tf.keras.losses.binary_crossentropy(y, predictions)
    gradients = tape.gradient(loss, model.trainable_weights)
    gradients = [g.numpy() for g in gradients]
    return gradients

def connect():
    try:
        response = requests.get('http://127.0.0.1:5000/get_connection')
        if response.status_code == 200:
            print("Connected to server successfully.")
            # Esegui ulteriori operazioni dopo la connessione
        else:
            print(f"Failed to connect to server. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def main():
    # creazione modello
    connect()
    print("hello")
    model = create_model()

    train_dataset, test_dataset = get_dataset_bosch()
    X_train, X_test = train_test_split(train_dataset, test_size=0.2, random_state=42)
    print(len("ddd"))





    """"
    #Generazione dati (ora sono randomici)
    #x_train = np.random.rand(10, 64, 64, 3)
    #y_train = np.random.randint(0, 2, size=(10, 1))
    
    for i in range(5):
        # calcolo dei pesi
        gradients = get_gradients(model, x_train, y_train)

        # invio dei pesi al boss del poppin
        data = {'gradients': [g.tolist() for g in gradients]}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(UPLOAD_URL, data=json.dumps(data), headers=headers)
        print('pesi inviati ihihihihihh')

        if response.status_code == 200:
            print('Gradients sent successfully')
        else:
            print('Failed to send gradients', response.text)

        
        updated_model_weights = download_updated_model()
        """

if __name__ == '__main__':
    main()
