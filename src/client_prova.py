import numpy as np
import tensorflow as tf
import requests
import json
from flask import Flask, request, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

app = Flask(__name__)

UPLOAD_URL = 'http://127.0.0.1:5000/upload_gradients'
MODEL_URL = 'http://127.0.0.1:5000/get_model'

# CREAZIONE MODELLO (UNIFORME SU TUTTI I CLIENT + SERVER)
def create_model():
    new_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    new_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    return new_model

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
    model = create_model()

    # Generazione dati (ora sono randomici)
    x_train = np.random.rand(10, 64, 64, 3)
    y_train = np.random.randint(0, 2, size=(10, 1))
    
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

if __name__ == '__main__':
    main()
