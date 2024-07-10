import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import requests
import io
import time

import random
from client_utils import create_model_and_dataset, train_model, validate_model, send_weights
from torch.utils.data import DataLoader

"""
    #definizione di batch_size e dei train e test loader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    #impostazione di criterior e optimizer da passare al modello
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Addestriamo il modello
    train_model(model, train_loader, criterion, optimizer, num_epochs=1)
    
    #validazionde del modello
    accuracy = validate_model(model, test_loader)

    # Salvataggio dei pesi del modello
    torch.save(model.state_dict(), './src/model_weights.pth')
    print("Model weights saved to 'model_weights.pth'")

    #model.load_state_dict(torch.load('./src/model_weights.pth'))
    #print(model.state_dict())

    #salvataggio pesi nella variabile
    model_weights = model.state_dict()
"""

def main():
    # Creazione modello e dei dataset
    model, train_dataset, test_dataset = create_model_and_dataset()

    ###############################
    model.load_state_dict(torch.load('./src/model_weights.pth'))
    model_weights = model.state_dict()
    ###############################


    #connessione al server, ed invio dell'id del client
    server_url_connect = ' http://127.0.0.1:5000/connect'  

    #L'id da inviare
    id = random.randint(1, 50)
    data_id = {'id': id}  

    # Invia la richiesta POST
    response = requests.get(server_url_connect, json=data_id)
    while(response.status_code != 200):
        id = random.randint(1, 50)
        data_id = {'id': id} 
        response = requests.post(server_url_connect, json=data_id)
        print(response.status_code)
        print(response.text)
    print(response.text)
    print(response.status_code)
    print('iddd ====== ', id)
    ################################INVIO PESI AL SERVER##########################################
    accuracy_rand = random.randint(1, 60)
    send_weights(model_weights, accuracy_rand, id)
    ##############################################################################################

    ######################RICEVI I NUOVI PESI DAL MODELLO CENTRALE################################

    #URL del server per ricevere i pesi
    server_url_load = 'http://localhost:5000/get_permission'

    #addestramento in 10 epoche 
    epoch = 10
    for i in range(epoch):
        print('EPOCA ------------------>', i)

        response = requests.get(server_url_load, json=data_id)
        data = response.json()
        codice = data['codice']
        print(response.text)

        while codice == 2:      #DEVO ATTENDERE GLI ALTRI CLIENT
            time.sleep(5)
            response = requests.get(server_url_load, json=data_id)
            data = response.json()
            codice = data['codice']
            print(response.text)
        if codice == 1:         #POSSO PROCEDERE CON I NUOVI PESI
            response = requests.get('http://localhost:5000/get_model_weight')
            buffer = io.BytesIO(response.content)

            # Deserializza il tensore
            tensor = torch.load(buffer)

            print("Tensore ricevuto:")
            print(tensor)
            print(type(tensor))
            model_weights = tensor
        if codice == 0:         #POSSO PROCEDERE CON I VECCHI PESI
            print(response.text)

       

        model.load_state_dict(model_weights)

        #train_model(model, train_loader, criterion, optimizer, num_epochs=1)

        #valutazione del modello
        #accuracy = validate_model(model, test_loader)

        ################################INVIO PESI AL SERVER##########################################
        send_weights(model_weights, random.randint(1,50), id)
        ##############################################################################################


    ##############################################################################################
        
if __name__ == "__main__":
    main()





