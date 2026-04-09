import os, cv2
import numpy as np
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, "video")

USE_WEBCAM = False

VEHICLE = [2, 5, 7]

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, "v.mp4")
    cap = cv2.VideoCapture(VIDEO_PATH)

model = YOLO("yolov8s.pt")

CONF_THRESHOLD = 0.3

RESIZE_WIDTH = 960

prev_time = time.time()
fps = 0.0

counter_frame = 0
count_frame = 0
count_frame_old = 0
count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        new_w = int(scale * w)
        new_h = int(scale * h)

        frame = cv2.resize(frame, (new_w, new_h))

    result = model(frame, conf = CONF_THRESHOLD, verbose=False)

    psevdo_id = 0

    PERSON_CLASS_ID = 0

    for r in result:
        count_frame = 0
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls in VEHICLE:
                psevdo_id += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                label = f"id: {psevdo_id} conf: {round(100*conf)}%"
                label2 = "color"
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                cv2.putText(frame, label2, (x1, y1-13), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                riz = int((y2-y1)/4)
                riz2 = int((x2-x1)/4)
                cv2.putText(frame, "#", (x1+45, y1-13), cv2.FONT_HERSHEY_PLAIN, 1, cv2.mean(frame[y1+riz:y2-riz, x1+riz2:x2-riz2])[:3], 3)
                if counter_frame==4:
                    if y2>=500:
                        count_frame+=1

        if counter_frame==4:
            counter_frame = 0
            if count_frame >= count_frame_old:
                count+=count_frame-count_frame_old
            count_frame_old = count_frame
        else:
            counter_frame+=1

    cv2.putText(frame, f"vehicle count: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255 ,0), 2)
    cv2.imshow("YOLO", frame)

    if cv2.waitKey(1) & 0xff == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()