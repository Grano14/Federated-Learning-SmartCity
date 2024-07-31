# Smart-City Federated-Learning 


Questo progetto implementa un sistema di Federated Learning nel contesto delle Smart City, in particolare la gestione dei semafori. Il Federated Learning permette ai dispositivi periferici di collaborare nell'addestramento di modelli di machine learning senza condividere i dati sensibili, garantendo così la privacy dei dati degli utenti.

## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision
```
pip install requirements.txt
```

## Running 
* Per eseguire il server che gestisce i pesi bisogna eseguire il comando seguente dalla directory ```./Federated-Learning-SmartCity/```

```
python src/server.py 
```

-----

* Per eseguire i client bisogna posizionarsi nella directory ```./Federated-Learning-SmartCity/yoloClient/``` e si troveranno 5 cartelle, una per ogni client. Bisogna lanciare il comando seguente 5 volte per eseguire i client posizionandosi nelle cartelle dei client:
```
python train.py --img 720 --batch 4 --epochs 1 --data ./dataset/dataset.yaml --cfg models/yolov5s.yaml --name yolov5_traffic_lights
```

## Results

* I risultati saranno stampati sulla console dei singoli client. 