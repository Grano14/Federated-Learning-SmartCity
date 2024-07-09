from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

app = Flask(__name__)

# Initialize central model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=SGD(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

central_model = create_model()

# Store gradients from clients
client_gradients = []

# Parameters for AFO
update_frequency = 1  # Update frequency in rounds
current_round = 0

@app.route('/upload_gradients', methods=['POST'])
def upload_gradients():
    global client_gradients, current_round
    data = request.json
    print("Data received:", data)  # Log the received data
    gradients = np.array(data['gradients'])
    client_gradients.append(gradients)
    
    # Check if we have gradients from all clients
    if len(client_gradients) == 6:
        # Average gradients (FedAvg)
        average_gradients = np.mean(client_gradients, axis=0)
        
        # Apply gradients to the central model
        apply_gradients(average_gradients)
        
        # Clear gradients for next round
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
