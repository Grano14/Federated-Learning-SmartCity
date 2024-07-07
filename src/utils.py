#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from datasets import load_dataset

from torch.utils.data import Dataset, random_split
import yaml
from PIL import Image


#CODICE PER ACCESSO REMOTO AL DATASET
def get_dataset_remote(args):
     
    if args.dataset == 'bosch':  #RICORDATE DI CAMBIARE L'ARGOMENTO DA LINEA DI COMANDO QUANDO ESEGUITE 
        print("sto qua")
        data_files = {"train": "train.tar.gz", "test": "test.tar.gz"}
        dataset = load_dataset("shpotes/bosch-small-traffic-lights-dataset", trust_remote_code=True, revision="main", data_files=data_files)
        print("dataset scaricato")
        print(dataset)
        data_dir = '../data/dataset/'
        train_data = dataset['train']
        test_data = dataset['test']

        #ISTRUZIONE PER CARICARE SOLO UN FILE DEL DATASET
        #subset = load_dataset("shpotes/bosch-small-traffic-lights-dataset", data_files="train.tar.gz")

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)
        return train_dataset, test_dataset, user_groups

#funione per ottenere il dataset dal file locale 
def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    #classe per la lettura del file train.yaml che contiene le label delle immagini e per l'istanziazione del dataset
    class TrafficLightDataset(Dataset):
        def __init__(self, yaml_file, transform=None):
            with open(yaml_file, 'r') as file:
                self.data = yaml.safe_load(file)
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_path = self.data[idx]['path']
            image = Image.open(img_path).convert('RGB')

            boxes = self.data[idx]['boxes']
            labels = []
            for box in boxes:
                labels.append(box['label'])

            if self.transform:
                image = self.transform(image)

            # If multi-label classification is needed, convert to a one-hot encoding or similar representation
            #label di test per il funzionamento del modello (va aggiunta la logica di mapping)
            #funzione di mapping
            label = 00000
            if 'Green' in labels:
                label += 1000
            if 'Yellow' in labels:
                label += 1000
            if 'Red' in labels:
                label += 100
            if 'off' in labels:
                label += 10

            labels = label
            print(labels)

            return image, labels

    #istanziazione della variabile transform per andare a trasformare le immagini 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if args.dataset == 'bosch':

        #inserire il path del file train.yaml
        #istanziazione del dataset
        dataset = TrafficLightDataset(yaml_file='/home/giuseppe/ProgettoSmartCity/fdsml/venv/src/train.yaml', transform=transform)

        #instanziazione valore per lo split del dataset
        test_ratio = 0.2  # 20% dei dati per il set di test
        num_total = len(dataset)
        num_test = int(test_ratio * num_total)
        num_train = num_total - num_test

        # Dividiamo il dataset in set di addestramento e di test
        train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

        #stampa numero di istanze nel trainset e testset
        print("Numero di campioni nel set di addestramento:", len(train_dataset))
        print("Numero di campioni nel set di test:", len(test_dataset))

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


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


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
