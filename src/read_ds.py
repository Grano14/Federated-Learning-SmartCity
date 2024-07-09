import yaml

# Leggere il file YAML
with open('./dataset/train/train.yaml', 'r') as file:
    data = yaml.safe_load(file)

new_text = '['

# Visualizzare il contenuto del file YAML
for entry in data:
    
    if entry['boxes']:
        for box in entry['boxes']:
            new_text = new_text + '{ \n\t "path": "' + entry['path'] + '", \n\t "boxes": {\n\t\t"label": "' + box['label'] + '",\n\t\t "x_min": ' + str(box['x_min']) + ',\n\t\t "x_max": ' + str(box['x_max']) + ',\n\t\t "y_min":' + str(box['y_min']) + ',\n\t\t "y_max":' + str(box['y_max']) + '\n\t}\n},\n'
            #print(f"  Label: {box['label']}, Occluded: {box['occluded']}, Coordinates: ({box['x_min']}, {box['y_min']}) to ({box['x_max']}, {box['y_max']})")
    else: 
        new_text = new_text + '{ \n\t "path": "' + entry['path'] + '", \n\t "boxes": { }\n},' + '\n'
new_text = new_text[:-2]
new_text = new_text + ']'

# Apriamo il file in modalità append (aggiunta)
with open('./train.txt', 'a') as file:
    file.write(new_text + '\n')
print(new_text)

