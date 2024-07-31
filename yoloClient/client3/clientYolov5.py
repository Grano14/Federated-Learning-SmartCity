import torch
import requests
import io
import time
import argparse
import random
from client_utils import create_model_and_dataset, train_model, validate_model, send_weights
from torch.utils.data import DataLoader
import subprocess
import sys
import os
from models.yolo import Model
from readDataset import CustomMulticlassDataset, evaluate_model, calculate_metrics
from torchvision import transforms
import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from utils.torch_utils import (
        EarlyStopping,
        ModelEMA,
        de_parallel,
        select_device,
        smart_DDP,
        smart_optimizer,
        smart_resume,
        torch_distributed_zero_first,
    )


def main():
    """
    # Creazione del parser
    parser = argparse.ArgumentParser(description='Client Federated Learning')

    # Aggiunta degli argomenti
    parser.add_argument('--dataset', type=str, help='Seleziona il dataset da utilizzare')

    # Parsing degli argomenti
    args = parser.parse_args()

    dataset_str = ''
    if args.dataset:
        if args.dataset == 'dataset1':
            dataset_str = './dataset/datasetFile/dataset1.json'
        if args.dataset == 'dataset2':
            dataset_str = './dataset/datasetFile/dataset2.json'
        if args.dataset == 'dataset3':
            dataset_str = './dataset/datasetFile/dataset3.json'
        if args.dataset == 'dataset4':
            dataset_str = './dataset/datasetFile/dataset4.json'
        if args.dataset == 'dataset15':
            dataset_str = './dataset/datasetFile/dataset5.json'
        if args.dataset == 'dataset6':
            dataset_str = './dataset/datasetFile/dataset6.json'
        if args.dataset == 'dataset7':
            dataset_str = './dataset/datasetFile/dataset7.json'
        if args.dataset == 'dataset8':
            dataset_str = './dataset/datasetFile/dataset8.json'
        if args.dataset == 'dataset9':
            dataset_str = './dataset/datasetFile/dataset9.json'
    else:
        print("dataset non selezionato")

"""
    # Addestriamo e validiamo il modello
    command = 'python3 train.py --img 416 --batch 4 --epochs 10 --data ./dataset/dataset.yaml --cfg models/yolov5s.yaml --name yolov5_traffic_lights'

    try:
        # Esegui il comando utilizzando subprocess.run con shell=True
        result = subprocess.run(command, capture_output=True, text=True, shell=True, check=True)

        # Stampa l'output del secondo script
        #print("Output del secondo script:")
        #print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Errore durante l'esecuzione del comando:", e)
        print("Output di errore:", e.stderr)

    #carica contenuto del file creato da yolo
    model_file_path = './pesi.pt'
    model_info = torch.load(model_file_path, map_location="cpu")
    #print(model_info['model'])

    with open("./f1score.txt", "r") as file:
        # Leggere il contenuto del file
        contenuto = file.read()

    f1score = contenuto

    #get model
    LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
    model = Model(model_info["model"].yaml).to("cpu")
    #print(model.state_dict())

    #get weights
    model_weights = model.state_dict()


    RANK = int(os.getenv("RANK", -1))
    ema = ModelEMA(model) if RANK in {-1, 0} else None
    modello = deepcopy(de_parallel(model)).half()
    ema = deepcopy(ema.ema).half()

    model_info['model'] = modello
    #print(model_info['model'])



    ##########################connessione al server, ed invio dell'id del client###################
    #server url
    server_url_connect = ' http://127.0.0.1:5000/connect'

    # L'id da inviare
    id = random.randint(1, 50)
    data_id = {'id': id}

    # Invia la richiesta POST
    response = requests.get(server_url_connect, json=data_id)
    while (response.status_code != 200):
        id = random.randint(1, 50)
        data_id = {'id': id}
        response = requests.post(server_url_connect, json=data_id)
        # print(response.status_code)
        # print(response.text)
    # print(response.text)
    # print(response.status_code)
    print('id ======> ', id)
    ##############################################################################################

    with open("./f1score.txt", "r") as file:
        # Leggere il contenuto del file
        contenuto = file.read()

    f1score = contenuto

    #invio pesi al server
    accuracy_rand = random.randint(1, 60)
    send_weights(model_weights, f1score, id)

    # URL del server per ricevere i pesi
    server_url_load = 'http://127.0.0.1:5000/get_permission'

    #addestramento in 10 epoche
    epoch = 10
    for i in range(epoch):
        print('EPOCA ------------------>', i, 'f1score', f1score)

        response = requests.get(server_url_load, json=data_id)
        data = response.json()
        codice = data['codice']
        # print(response.text)

        while codice == 2:  # DEVO ATTENDERE GLI ALTRI CLIENT
            time.sleep(5)
            response = requests.get(server_url_load, json=data_id)
            data = response.json()
            codice = data['codice']
            # print(response.text)
        if codice == 1:  # POSSO PROCEDERE CON I NUOVI PESI
            response = requests.get('http://127.0.0.1:5000/get_model_weight')
            buffer = io.BytesIO(response.content)

            # Deserializza il tensore
            tensor = torch.load(buffer)

            #assegna i nuovi pesi ricevuti
            model_weights = tensor
        # if codice == 0:         #POSSO PROCEDERE CON I VECCHI PESI
        # print(response.text)

        #caricamento pesi sul modello
        model.load_state_dict(model_weights)

        #creare il file pesi.pt, viene utilizzato da yolo
        RANK = int(os.getenv("RANK", -1))
        ema = ModelEMA(model) if RANK in {-1, 0} else None
        modello = deepcopy(de_parallel(model)).half()
        ema = deepcopy(ema.ema).half()
        #print(model_info['model'])
        model_info['model'] = modello
        #print(model_info['model'])

        torch.save(model_info, './update_pesi.pt')

        # Addestriamo e validiamo il modello
        new_command = 'python3 train.py --img 416 --batch 4 --epochs 10 --data ./dataset/dataset.yaml --cfg models/yolov5s.yaml --name yolov5_traffic_lights --weights ./update_pesi.pt'

        try:
            # Esegui il comando utilizzando subprocess.run con shell=True
            result = subprocess.run(new_command, capture_output=True, text=True, shell=True, check=True)

            # Stampa l'output del secondo script
            #print("Output del secondo script:")
            #print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("Errore durante l'esecuzione del comando:", e)
            print("Output di errore:", e.stderr)

        # carica contenuto del file creato da yolo
        model_file_path = './pesi.pt'
        model_info = torch.load(model_file_path, map_location="cpu")
        # print(model_info['model'])

        # get model
        LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
        model = Model(model_info["model"].yaml).to("cpu")
        # print(model.state_dict())

        # get weights
        model_weights = model.state_dict()

        with open("./f1score.txt", "r") as file:
            # Leggere il contenuto del file
            contenuto = file.read()

        f1score = contenuto

        ################################INVIO PESI AL SERVER##########################################
        accuracy_rand = random.randint(1, 60)
        send_weights(model_weights, f1score, id)
        ##############################################################################################

    ##############################################################################################

if __name__ == "__main__":
    main()
