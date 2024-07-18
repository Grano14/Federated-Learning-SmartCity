import yaml
import json

""""
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


with open('./dataset/test/test.yaml', 'r') as file:
    data = yaml.safe_load(file)

new_text = '['

for entry in data:
    entry['path'] = entry['path'].replace('/net/pal-soc1.us.bosch.com/ifs/data/Shared_Exports/deep_learning_data/traffic_lights/university_run1/', './dataset/test/rgb/test/')
    
    if entry['boxes']:
        for box in entry['boxes']:
            new_text = new_text + '{ \n\t "path": "' + entry['path'] + '", \n\t "boxes": {\n\t\t"label": "' + box['label'] + '",\n\t\t "x_min": ' + str(box['x_min']) + ',\n\t\t "x_max": ' + str(box['x_max']) + ',\n\t\t "y_min":' + str(box['y_min']) + ',\n\t\t "y_max":' + str(box['y_max']) + '\n\t}\n},\n'
            #print(f"  Label: {box['label']}, Occluded: {box['occluded']}, Coordinates: ({box['x_min']}, {box['y_min']}) to ({box['x_max']}, {box['y_max']})")
    else: 
        new_text = new_text + '{ \n\t "path": "' + entry['path'] + '", \n\t "boxes": { }\n},' + '\n'
new_text = new_text[:-2]
new_text = new_text + ']'

with open('./test.json', 'a') as file:
    file.write(new_text + '\n')
print(new_text)


with open('/home/giuseppe/ProgettoSmartCity/fdsml/venv/src/Federated-Learning-PyTorch-master/dataset/test/test.json', 'r') as file:
    data = json.load(file)

new_text1 = '['
new_text2 = '['
new_text3 = '['
new_text4 = '['
new_text5 = '['

i = 0

for item in data:
    if i < 2700:
        new_text1 += str(item) + ',\n'
    if i >= 2700 and i < 5400:
        new_text2 += str(item) + ',\n'
    if i >= 5400 and i < 8100:
        new_text3 += str(item) + ',\n'
    if i >= 8100 and i < 10800:
        new_text4 += str(item) + ',\n'   
    if i >= 10800 and i < 13500:
        new_text5 += str(item) + ',\n' 
    i = i + 1 

new_text1 = new_text1[:-2]
new_text2 = new_text2[:-2]
new_text3 = new_text3[:-2]
new_text4 = new_text4[:-2]
new_text5 = new_text4[:-2]

new_text1 += ']'
new_text2 += ']'
new_text3 += ']'
new_text4 += ']'
new_text5 += ']'

with open('./dataset5.json', 'a') as file:
    file.write(new_text1 + '\n')
with open('./dataset6.json', 'a') as file:
    file.write(new_text2 + '\n')
with open('./dataset7.json', 'a') as file:
    file.write(new_text3 + '\n')
with open('./dataset8.json', 'a') as file:
    file.write(new_text4 + '\n')
with open('./dataset9.json', 'a') as file:
    file.write(new_text5 + '\n')
"""


with open('/home/giuseppe/ProgettoSmartCity/fdsml/venv/src/Federated-Learning-PyTorch-master/dataset/datasetFile/dataset9.json', 'r') as file:
    data = json.load(file)
yellow, red, green, off = 0, 0, 0, 0
for item in data:
    if item['boxes']:
        if item['boxes']['label'] == 'Red':
            red += 1
        if item['boxes']['label'] == 'Green':
            green += 1
        if item['boxes']['label'] == 'Yellow':
            yellow += 1
        if item['boxes']['label'] == 'off':
            off += 1

print(red)
print(green)
print(yellow)
print(off)
        

