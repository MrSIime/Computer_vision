import cv2
import numpy as np  
import os

file_names = os.listdir("images")

face_cascade = cv2.CascadeClassifier('C:/Users/Student/Desktop/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/Student/Desktop/data/haarcascades/haarcascade_eye.xml')

face_net = cv2.dnn.readNetFromCaffe('C:/Users/Student/Desktop/data/DNN/deploy.prototxt','C:/Users/Student/Desktop/data/DNN/res10_300x300_ssd_iter_140000.caffemodel')

os.makedirs("images", exist_ok=True)
os.makedirs("out_images", exist_ok=True)

for file_name in file_names:

    if not file_name.split(".")[-1] in ["jpg", "png", "tiff", "webp", "jpeg"]:
        continue

    cap = cv2.VideoCapture(f"images/{file_name}")

    ret, frame = cap.read()

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30, 30)
    )

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
                
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")

            x, y = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    for (x, y, w, h) in faces:

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.1,minNeighbors = 10, minSize = (15, 15))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)


    cv2.imwrite(f"out_images/{(file_name.split("."))[0]}_out.jpg", frame)

