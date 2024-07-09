import cv2
import numpy as np
import os

# Percorsi ai file di configurazione e peso
cfg_path = '/home/giuseppe/ProgettoSmartCity/fdsml/venv/src/Federated-Learning-PyTorch-master/src/yolo/darknet/cfg/yolov3.cfg'
weights_path = '/home/giuseppe/ProgettoSmartCity/fdsml/venv/src/Federated-Learning-PyTorch-master/src/yolo/darknet/yolov3.weights'
names_path = '/home/giuseppe/ProgettoSmartCity/fdsml/venv/src/Federated-Learning-PyTorch-master/src/yolo/darknet/data/coco.names'

# Verifica che i file esistano
if not os.path.exists(cfg_path):
    raise FileNotFoundError(f"File di configurazione non trovato: {cfg_path}")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"File di pesi non trovato: {weights_path}")
if not os.path.exists(names_path):
    raise FileNotFoundError(f"File di nomi delle classi non trovato: {names_path}")

# Load YOLOv3 configuration and weights
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load COCO class names
with open(names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# Define the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_objects(image):
    height, width = image.shape[:2]
    
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Run forward pass to get the network output
    layer_outputs = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    
    # Process each output
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter out weak detections
            if confidence > 0.5 and classes[class_id] == "traffic light":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    result_boxes = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        result_boxes.append(box)
        
    return result_boxes

# Load the input image
image_path = '/home/giuseppe/ProgettoSmartCity/fdsml/venv/src/Federated-Learning-PyTorch-master/dataset/train/rgb/train/2015-10-05-10-52-01_bag/27930.png'
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Immagine non trovata: {image_path}")

# Detect objects in the image
traffic_lights = detect_objects(image)

# Draw bounding boxes around detected traffic lights and extract them
for idx, (x, y, w, h) in enumerate(traffic_lights):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Extract the traffic light
    traffic_light = image[y:y+h, x:x+w]
    
    # Save the extracted traffic light with a unique name
    cv2.imwrite(f'traffic_light_{idx}.jpg', traffic_light)
    
# Display the output image with detected traffic lights
cv2.imshow('Detected Traffic Lights', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
