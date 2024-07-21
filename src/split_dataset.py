import os
import shutil
import math

# Percorsi delle cartelle
cartella_txt = r'C:\Users\gioac\Desktop\FDSML\progetto\Federated-Learning-SmartCity\dataset\LABELS\train'
cartella_immagini = r'C:\Users\gioac\Desktop\FDSML\progetto\Federated-Learning-SmartCity\dataset\dataset filtrato\train'
cartella_destinazione_immagini = r'C:\Users\gioac\Desktop\FDSML\progetto\Federated-Learning-SmartCity\dataset\dataset diviso\train'
cartella_destinazione_txt = r'C:\Users\gioac\Desktop\FDSML\progetto\Federated-Learning-SmartCity\dataset\dataset diviso\train'

# Crea le 6 cartelle di destinazione per le immagini
num_cartelle = 6
for i in range(1, num_cartelle + 1):
    os.makedirs(os.path.join(cartella_destinazione_immagini, f'cartella_{i}'), exist_ok=True)
    os.makedirs(os.path.join(cartella_destinazione_txt, f'cartella_{i}'), exist_ok=True)

# Ottieni i nomi dei file .txt senza estensione
nomi_txt = [os.path.splitext(f)[0] for f in os.listdir(cartella_txt) if f.endswith('.txt')]

# Ordina i nomi per avere una divisione più bilanciata
nomi_txt.sort()

# Dividi i nomi in 6 gruppi bilanciati
num_per_cartella = math.ceil(len(nomi_txt) / num_cartelle)
gruppi_nomi = [nomi_txt[i:i + num_per_cartella] for i in range(0, len(nomi_txt), num_per_cartella)]

# Copia i file nelle rispettive cartelle
for i, gruppo in enumerate(gruppi_nomi):
    for nome in gruppo:
        # Copia l'immagine
        for estensione in ['.jpg', '.png']:  # Aggiungi altre estensioni se necessario
            immagine_src = os.path.join(cartella_immagini, nome + estensione)
            if os.path.exists(immagine_src):
                immagine_dst = os.path.join(cartella_destinazione_immagini, f'cartella_{i + 1}', nome + estensione)
                shutil.copy(immagine_src, immagine_dst)
                break

        # Copia il file .txt
        txt_src = os.path.join(cartella_txt, nome + '.txt')
        txt_dst = os.path.join(cartella_destinazione_txt, f'cartella_{i + 1}', nome + '.txt')
        shutil.copy(txt_src, txt_dst)
        
        print(f'Copiato: {nome} in cartella_{i + 1}')

print('Operazione completata.')



