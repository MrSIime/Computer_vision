import cv2
import os
from ultralytics import YOLO
import yt_dlp


PROJECT_DIR = os.path.dirname(__file__)
# VIDEO_dir = os.path.join()

YOUTUBE = 'https://www.youtube.com/watch?v=M3EYAY2MftI'
MODEL_PATH = 'yolov8n.pt'

track_history = {}
PPM = 8


def get_stream_url(url):
    ydl_opts = {
        'format': 'bestvideo[height<=480][ext=mp4]/best[height<=480]/worst',
        'quiet': True,
        'no_warnings': True,
    }
    print("📡 З'єднання з YouTube через yt-dlp...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except Exception as e:
        print(f" Помилка: {e}")
        return None


model = YOLO(MODEL_PATH)
stream = get_stream_url(YOUTUBE)
if not stream:
    exit()

cap = cv2.VideoCapture(stream)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.track(
        frame,
        classes=[2],
        conf=0.5,
        verbose=False
    )
    if result[0].boxes.id is not None:
        boxes = result[0].boxes.xywh.cpu().numpy()
        tracks_ids = result[0].boxes.id.int().cpu().tolist()

        for box, trackid in zip(boxes, tracks_ids):
            x, y, w, h = box

            if trackid in track_history:
                prev_x, prev_y = track_history[trackid]
                dist_px = ((x - prev_x) ** 2 + (y - prev_y) ** 2) ** 0.5
                speed = (dist_px / PPM) * fps
                speed_km = speed * 3.6

                cv2.putText(frame, f"id - {id} speed - {speed_km}", (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                
    car_frame = result[0].plot()

    frame_count += 1
    cv2.imshow('stream', car_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()