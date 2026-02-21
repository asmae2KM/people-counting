'''en utilisant le mask
import cvzone
from ultralytics import YOLO
import cv2
import math
import numpy as np
from sort import *
import matplotlib.pyplot as plt



#cap = cv2.VideoCapture(2)
cap = cv2.VideoCapture("C:/Users/DELL/Downloads/DeepLearning/Video.mp4")
#model = YOLO("best.pt")
model_path = 'runs/detect/train/weights/best.pt'  # Remplacez par le chemin de votre modèle
model = YOLO(model_path)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask1 = cv2.imread("mask.png")

tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)

limitsDown = [510, 150, 780, 150]
limitsUp = [10, 350, 250, 350]

totalCountUp = []
totalCountDown = []

while True:
    success, img = cap.read()
    # Redimensionner le masque pour qu'il corresponde à la taille de l'image
    mask = cv2.resize(mask1, (img.shape[1], img.shape[0]))
    imgRegion = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (700, 360))
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.4:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                scale = 0.6, thickness = 1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=7, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    trackerResults = tracker.update(detections)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)

    for result in trackerResults:
        x1, y1, x2, y2 , id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=7, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale = 0.6, thickness = 1, offset=3)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 30 < cy < limitsDown[1] + 30:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)


    for result in trackerResults:
        x1, y1, x2, y2 , id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=7, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale = 0.6, thickness = 1, offset=3)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 30 < cy < limitsUp[1] + 30:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

    #cvzone.putTextRect(img, f' Car Counter:{len(totalCount)}', (50, 50))

    cv2.putText(img, str(len(totalCountDown)), (1145, 450), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 7)
    cv2.putText(img, str(len(totalCountUp)), (870, 450), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 7)


    # Dans la boucle while
    
    #cv2.imshow("Webcam", img)
    #cv2.imshow("Mask", imgRegion)

    #key = cv2.waitKey(1)
    #if key == ord('q'):
     #   break
    #if key == ord('p'):
     #   cv2.waitKey(-1)
    # if cv2.waitKey(1) & 0xff == ord('q'):
    #     break

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
#cap.release()
#cv2.destroyAllWindows()
'''
'''
import cv2
from ultralytics import YOLO
import numpy as np
from sort import *
import matplotlib.pyplot as plt
import math

cap = cv2.VideoCapture("C:/Users/DELL/Downloads/DeepLearning/Video.mp4")
model_path = 'runs/detect/train/weights/best.pt'
model = YOLO(model_path)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
totalCountUp = []
totalCountDown = []

limitsDown = [510, 150, 780, 150]
limitsUp = [10, 350, 250, 350]

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.4:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    trackerResults = tracker.update(detections)

    for result in trackerResults:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cx, cy = x1 + w // 2, y1 + h // 2

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 30 < cy < limitsDown[1] + 30:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

    for result in trackerResults:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cx, cy = x1 + w // 2, y1 + h // 2

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 30 < cy < limitsUp[1] + 30:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

    cv2.putText(img, str(len(totalCountDown)), (1145, 450), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 7)
    cv2.putText(img, str(len(totalCountUp)), (870, 450), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 7)

if not img.size == 0:
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.show()
'''
"""
import cvzone
from ultralytics import YOLO
import cv2
import numpy as np
from sort import *
import matplotlib.pyplot as plt

# Charger le modèle YOLO
model_path = 'runs/detect/train/weights/best.pt'
model = YOLO(model_path)

# Initialiser le tracker SORT
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Définir les classes YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Définir les limites pour le comptage
limitsDown = [510, 150, 780, 150]
limitsUp = [10, 350, 250, 350]

# Ouvrir la vidéo
cap = cv2.VideoCapture("C:/Users/DELL/Downloads/DeepLearning/Video.mp4")

# Définir les paramètres pour écrire la vidéo de sortie
out_filename = 'video_output.mp4'
codec = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Créer un objet VideoWriter pour écrire la vidéo de sortie
out = cv2.VideoWriter(out_filename, codec, fps, (frame_width, frame_height))

# Boucle principale de traitement des images
while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Effectuer la détection YOLO
    results = model(img, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            w, h = int(x2 - x1), int(y2 - y1)
            conf = round(box.conf[0].item(), 2)
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Filtrer les détections de personnes avec une certaine confiance
            if currentClass == "person" and conf > 0.4:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Mettre à jour le tracker SORT
    trackerResults = tracker.update(detections)

    # Compter le nombre de personnes dans chaque zone
    totalCountUp = sum(1 for result in trackerResults if
                      limitsUp[0] < result[0] + result[2] / 2 < limitsUp[2] and
                      limitsUp[1] < result[1] + result[3] / 2 < limitsUp[3])

    totalCountDown = sum(1 for result in trackerResults if
                        limitsDown[0] < result[0] + result[2] / 2 < limitsDown[2] and
                        limitsDown[1] < result[1] + result[3] / 2 < limitsDown[3])

    # Dessiner les résultats sur l'image
    for result in trackerResults:
        x1, y1, x2, y2, id = result
        w, h = int(x2 - x1), int(y2 - y1)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(img, f'ID: {int(id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Dessiner les lignes de comptage
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)

    # Afficher le nombre de personnes dans chaque zone
    cv2.putText(img, str(totalCountDown), (1145, 450), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 7)
    cv2.putText(img, str(totalCountUp), (870, 450), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 7)

    # Écrire l'image traitée dans la vidéo de sortie
    out.write(img)

    # Afficher l'image
    #cv2.imshow("Output", img)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break
if not img.size == 0:
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

# Libérer les ressources
cap.release()
out.release()
#cv2.destroyAllWindows()
print(f"Évaluation terminée. La vidéo de sortie est sauvegardée sous '{out_filename}'.")
"""
import cv2
from ultralytics import YOLO
import numpy as np
from sort import Sort

# Charger le modèle YOLO
model_path = 'runs/detect/train/weights/last.pt'
model = YOLO(model_path)

# Initialiser le tracker SORT
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Définir les classes YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Définir les limites pour le comptage
limitsDown = [510, 150, 780, 150]
limitsUp = [10, 350, 250, 350]

# Ouvrir la vidéo
cap = cv2.VideoCapture("C:/Users/DELL/Downloads/DeepLearning/vedio.mp4")

# Vérifier si la vidéo est ouverte avec succès
if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la vidéo")
    exit()

# Définir les paramètres pour écrire la vidéo de sortie
out_filename = 'out_v2.mp4'
codec = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Créer un objet VideoWriter pour écrire la vidéo de sortie
out = cv2.VideoWriter(out_filename, codec, fps, (frame_width, frame_height))

# Boucle principale de traitement des images
while cap.isOpened():
    count = 0
    success, img = cap.read()
    if not success:
        break

    # Effectuer la détection YOLO
    results = model(img, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            w, h = int(x2 - x1), int(y2 - y1)
            conf = round(box.conf[0].item(), 2)
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Filtrer les détections de personnes avec une certaine confiance
            if currentClass == "person" and conf > 0.4:
                count += 1
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Mettre à jour le tracker SORT
    trackerResults = tracker.update(detections)

    # Compter le nombre de personnes dans chaque zone
    totalCountUp = sum(1 for result in trackerResults if
                      limitsUp[0] < result[0] + result[2] / 2 < limitsUp[2] and
                      limitsUp[1] < result[1] + result[3] / 2 < limitsUp[3])

    totalCountDown = sum(1 for result in trackerResults if
                        limitsDown[0] < result[0] + result[2] / 2 < limitsDown[2] and
                        limitsDown[1] < result[1] + result[3] / 2 < limitsDown[3])

    # Dessiner les résultats sur l'image
    for result in trackerResults:
        x1, y1, x2, y2, id = result
        w, h = int(x2 - x1), int(y2 - y1)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(img, f'ID: {int(id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Dessiner les lignes de comptage
    #cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)
    #cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)

    # Afficher le nombre de personnes dans chaque zone
    cv2.putText(img, str(totalCountDown), (1145, 450), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 7)
    cv2.putText(img, str(totalCountUp), (870, 450), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 7)
    cv2.putText(img, f'number of persons : {count}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    # Écrire l'image traitée dans la vidéo de sortie
    out.write(img)

# Libérer les ressources
cap.release()
out.release()
print(f"Évaluation terminée. La vidéo de sortie est sauvegardée sous '{out_filename}'.")
