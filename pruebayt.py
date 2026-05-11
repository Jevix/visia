import cv2
import subprocess

youtube_url = "https://www.youtube.com/watch?v=butK9aqBY1E"

stream_url = subprocess.check_output(
    ["yt-dlp", "-g", youtube_url],
    text=True
).strip().splitlines()[-1]

print("STREAM:", stream_url[:120], "...")

cap = cv2.VideoCapture(stream_url)
while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer stream")
        break

    cv2.imshow("VISIA YouTube", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()