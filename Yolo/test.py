#Importar librerias
print("Cargando librerias...")
import cv2
from pytorchyolo import detect, models

def cargarClases(path):
    clases=[]
    f = open(path,'r')
    for c in f:
        clases.append(c.replace('\n',''))
    f.close()
    return clases

def mostrarBoundingBox(frame,deteccion):
    for x1, y1, x2, y2, cls_conf, cls_pred in deteccion:
        if cls_pred!=0 and cls_pred!=39:
            break
        x1=int(x1)
        y1=int(y1)
        x2=int(x2)
        y2=int(y2)

        box_w = x2 - x1
        box_h = y2 - y1    

        frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color_borde, 2)
        cv2.putText(frame, clases[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color_letra, 2)# Nombre de la clase detectada
        cv2.putText(frame, str("%.2f" % float(cls_conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color_letra, 2) # Certeza de prediccion de la clase

    return frame

#Cargar red
print("Cargando red...")
clases=cargarClases('data/coco.names')
#model = models.load_model("./config/yolov3.cfg","./weights/yolov3.weights")
model = models.load_model("./config/yolov3Tiny.cfg","./weights/yolov3Tiny.weights")


#Empezar video
color_borde=[150,30,30]
color_letra=[0,100,100]
cap = cv2.VideoCapture(0)

print('Grabando Camra...')

while cap:
    ret, frame = cap.read()
    if ret is False:
        break

    #load image
    #frame = cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_CUBIC)

    #bgr to rgb
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    #Run YOLO
    deteccion = detect.detect_image(model,frame) # [[x1, y1, x2, y2, confidence, class]]
    
    frame=mostrarBoundingBox(frame,deteccion)

    cv2.imshow('frame',cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

