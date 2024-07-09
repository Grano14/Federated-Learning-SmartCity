import numpy as np
import tensorflow as tf
import requests
import json

SERVER_URL = 'http://127.0.0.1:5000/upload_gradients'

# Define the local model (should be same architecture as central model)
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_gradients(model, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = tf.keras.losses.binary_crossentropy(y, predictions)
    gradients = tape.gradient(loss, model.trainable_weights)
    gradients = [g.numpy() for g in gradients]
    return gradients

def main():
    # Create a local model
    local_model = create_model()

    # Generate dummy data
    x_train = np.random.rand(10, 64, 64, 3)
    y_train = np.random.randint(0, 2, size=(10, 1))

    # Calculate gradients
    gradients = get_gradients(local_model, x_train, y_train)

    # Send gradients to server
    data = {'gradients': [g.tolist() for g in gradients]}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(SERVER_URL, data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        print('Gradients sent successfully')
    else:
        print('Failed to send gradients', response.text)

if __name__ == '__main__':
    main()
