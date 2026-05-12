from ultralytics import YOLO
import cv2
import os
import time

MODEL_PATH = "yolov8n.pt"

# PC Linux Mint
VIDEO_PATH = "/home/jevix/Escritorio/cruce.mkv"

# Raspberry
# VIDEO_PATH = "/home/autotrol/cruce.mkv"

VEHICLE_CLASSES = [2, 3, 5, 7]

CONF = 0.35
IOU = 0.60
IMG_SIZE = 320
MIN_AREA = 700

RESIZE_WIDTH = 640
DETECT_EVERY = 3

# Espira virtual sobre frame reducido
ESPIRA = (40, 200, 230, 350)

FRAMES_ON = 2
FRAMES_OFF = 8

SHOW_VIDEO = True
REALTIME_PLAYBACK = True

print("Iniciando VISIA Pi3...")
print("Video:", VIDEO_PATH)

if not os.path.exists(VIDEO_PATH):
    print("ERROR: No existe el video")
    exit()

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: No se pudo abrir el video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps and fps > 0 else 33

print(f"FPS video: {fps:.2f}")
print(f"Delay base: {delay} ms")

detectando = 0
sin_detectar = 0
espira_activa = False
frame_count = 0
last_boxes = []
last_yolo_time = 0

while True:
    loop_start = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Fin del video")
        break

    frame_count += 1

    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / w
    new_h = int(h * scale)
    frame_small = cv2.resize(frame, (RESIZE_WIDTH, new_h))

    hay_vehiculo = False

    if frame_count % DETECT_EVERY == 0:
        t0 = time.time()

        results = model(
            frame_small,
            classes=VEHICLE_CLASSES,
            conf=CONF,
            iou=IOU,
            imgsz=IMG_SIZE,
            verbose=False
        )

        last_yolo_time = time.time() - t0
        last_boxes = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                area = (x2 - x1) * (y2 - y1)
                if area < MIN_AREA:
                    continue

                last_boxes.append((x1, y1, x2, y2, cls, conf))

    x1e, y1e, x2e, y2e = ESPIRA

    for x1, y1, x2, y2, cls, conf in last_boxes:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        dentro = x1e <= cx <= x2e and y1e <= cy <= y2e

        if dentro:
            hay_vehiculo = True

        if SHOW_VIDEO:
            color = (0, 255, 0) if dentro else (255, 255, 255)
            name = model.names[cls]

            cv2.rectangle(frame_small, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame_small, (cx, cy), 4, color, -1)
            cv2.putText(
                frame_small,
                f"{name} {conf:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

    if hay_vehiculo:
        detectando += 1
        sin_detectar = 0
    else:
        sin_detectar += 1
        detectando = 0

    if not espira_activa and detectando >= FRAMES_ON:
        espira_activa = True
        print("ESPIRA ON")

    if espira_activa and sin_detectar >= FRAMES_OFF:
        espira_activa = False
        print("ESPIRA OFF")

    if SHOW_VIDEO:
        color_espira = (0, 255, 0) if espira_activa else (0, 0, 255)

        cv2.rectangle(frame_small, (x1e, y1e), (x2e, y2e), color_espira, 2)

        cv2.putText(
            frame_small,
            f"ESPIRA: {'ON' if espira_activa else 'OFF'}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color_espira,
            2
        )

        cv2.putText(
            frame_small,
            f"Frame:{frame_count} YOLO:{last_yolo_time:.2f}s",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("VISIA Pi3", frame_small)

        if REALTIME_PLAYBACK:
            elapsed_ms = int((time.time() - loop_start) * 1000)
            wait_ms = max(1, delay - elapsed_ms)
        else:
            wait_ms = 1

        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()