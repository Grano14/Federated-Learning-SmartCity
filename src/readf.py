import json

# Nome del file JSON
nome_file = './train.json'

# Apriamo il file JSON in modalità lettura
with open(nome_file, 'r') as file:
    # Carichiamo i dati JSON
    data = json.load(file)

    list = []
    
    # Iteriamo sugli oggetti nel file JSON
    for oggetto in data:
        if('label' in oggetto['boxes']):
            if oggetto['boxes']['label'] not in list:
                list.append(oggetto['boxes']['label'])
    print(list)