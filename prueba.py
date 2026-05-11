from ultralytics import YOLO
import cv2
import time

# =========================
# CONFIGURACIÓN
# =========================

#RTSP_URL = "rtsp://admin:Claudio2022@192.168.100.50:554/cam/realmonitor?channel=2&subtype=1"
RTSP_URL = "rtsp://716f898c7b71.entrypoint.cloud.wowza.com:1935/app-8F9K44lJ/304679fe_stream2"
MODEL_PATH = "yolov8n.pt"   # después podés probar yolov8s.pt
CONF = 0.45
IMG_SIZE = 640

# Clases COCO
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Anti-parpadeo
FRAMES_PARA_ACTIVAR = 3
FRAMES_PARA_DESACTIVAR = 15

# Zona virtual: x1, y1, x2, y2
# Ajustala según tu imagen
ZONA = (0, 80, 100, 310)

# =========================
# INICIO
# =========================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

contador_detectando = 0
contador_sin_detectar = 0
salida_activa = False

def centro_bbox(x1, y1, x2, y2):
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy

def punto_en_zona(cx, cy, zona):
    zx1, zy1, zx2, zy2 = zona
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer frame")
        time.sleep(0.5)
        continue

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=VEHICLE_CLASSES,
        conf=CONF,
        imgsz=IMG_SIZE,
        verbose=False
    )

    hay_vehiculo_en_zona = False

    # Dibujar zona virtual
    zx1, zy1, zx2, zy2 = ZONA
    color_zona = (0, 255, 0) if salida_activa else (0, 0, 255)
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), color_zona, 2)
    cv2.putText(frame, "ZONA ESPIRA", (zx1, zy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_zona, 2)

    if results and results[0].boxes is not None:
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            cx, cy = centro_bbox(x1, y1, x2, y2)

            dentro = punto_en_zona(cx, cy, ZONA)

            if dentro:
                hay_vehiculo_en_zona = True

            color = (0, 255, 0) if dentro else (255, 255, 255)

            nombre = model.names[cls]
            texto = f"{nombre} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.putText(frame, texto, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # =========================
    # LÓGICA ANTI PARPADEO
    # =========================

    if hay_vehiculo_en_zona:
        contador_detectando += 1
        contador_sin_detectar = 0
    else:
        contador_sin_detectar += 1
        contador_detectando = 0

    if not salida_activa and contador_detectando >= FRAMES_PARA_ACTIVAR:
        salida_activa = True
        print("SALIDA ON - Vehículo detectado")

    if salida_activa and contador_sin_detectar >= FRAMES_PARA_DESACTIVAR:
        salida_activa = False
        print("SALIDA OFF - Zona libre")

    estado = "ON" if salida_activa else "OFF"
    cv2.putText(frame, f"ESPIRA: {estado}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if salida_activa else (0, 0, 255), 3)

    cv2.imshow("VISIA - Detector Vehicular Virtual", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()