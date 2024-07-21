import subprocess

def main():
    # Definisci i parametri da passare al secondo script
    param1 = "valore1"
    param2 = "valore2"

    command = 'python3 train.py --img 640 --batch 4 --epochs 1 --data ./dataset/dataset.yaml --cfg models/yolov5s.yaml --name yolov5_traffic_lights'

    try:
        # Esegui il comando utilizzando subprocess.run con shell=True
        result = subprocess.run(command, capture_output=True, text=True, shell=True, check=True)

        # Stampa l'output del secondo script
        print("Output del secondo script:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Errore durante l'esecuzione del comando:", e)
        print("Output di errore:", e.stderr)

if __name__ == "__main__":
    main()