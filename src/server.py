from flask import Flask
from flask import request, jsonify

app = Flask(__name__)

global media
global accuracy

@app.route('/update-model', methods=['POST'])
def update_model():
    global central_model   # !!!!BISOGNA DEFINIRE LA STRUTTURA DEL CENTRAL MODEL (LAYER DI OGNI CLIENT ECC)!!!!!
    
    central_model = {}

    #RICEZIONE PESI
    weights = request.json['weights']
    client_id = request.json['client_id']
    accuracy = request.json['accuracy']
    
    # AGGIORNAMENTO MODELLO CON I PESI RICEVUTI
    central_model[client_id] = weights
    
    #ESEMPIO DI AGGIORNAMENTO DEL MODELLO TRAMITE MEDIA DEI PESI
    if len(central_model) > 0:
        avg_weights = {}
        num_clients = len(central_model)
        for layer_name in central_model[client_id].keys():
            avg_weights[layer_name] = sum([central_model[client_id][layer_name] for client_id in central_model.keys()]) / num_clients
       
        central_model = avg_weights #ALLA FIN AGGIORNA IL VALORE DEL MODELLO CENTRALE CON I PESI 
    
    return jsonify({'message': 'modello centrale aggiornato con successo'})


@app.route('/get-model', methods=['GET'])
def get_model(client_id, central_model):
    """
    Ottiene il modello (o i pesi) associati al client_id da central_model.
    
    Args:
    - client_id (str): Identificativo del cliente
    - central_model (dict): Dizionario contenente i pesi dei modelli per ogni cliente e layer
    
    Returns:
    - dict or None: Dizionario dei pesi del modello per il client_id, se presente; None altrimenti
    """
    if client_id in central_model:
        return central_model[client_id]   #NON SO SE DOBBIAMO TORNARE SOLO I PESI AGGIORNATI DI QUEL CLIENT O I PESI TOTALI (CENTRAL_MODEL)
    else:
        return None

if __name__ == '__main__':
    app.run(debug=True)