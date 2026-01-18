import cv2
import numpy as np

img = cv2.imread('images/321.jpg')

scale = 2
img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))

img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5,5), 2)

img = cv2.equalizeHist(img)

img_edges = cv2.Canny(img, 50, 150)
kernel = np.ones((3,3), np.uint8)
img_edges = cv2.dilate(img_edges, kernel, iterations=1)

contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

found_contours = 0
for cnt in contours:
   
    area = cv2.contourArea(cnt)

    if area > 1000:
        found_contours += 1
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255,0,0), 2)

cv2.imwrite("out.jpg", img_copy)
print(f"Знайдено меж: {found_contours}")
