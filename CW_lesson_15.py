import os, cv2
import numpy as np
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, "video")
OUT_DIR = os.path.join(PROJECT_DIR, "out")

os.makedirs(OUT_DIR, exist_ok=True)

USE_WEBCAM = True

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, "name")
    cap = cv2.VideoCapture(VIDEO_PATH)

model = YOLO("yolov8s.pt")

CONF_THRESHOLD = 0.5

RESIZE_WIDTH = 960

prev_time = time.time()
fps = 0.0

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

    people_count = 0
    psevdo_id = 0

    PERSON_CLASS_ID = 0