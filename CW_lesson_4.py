import cv2
import numpy as np

img = cv2.imread('images/image1.png')

scale = 1
img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))

img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5,5), 2)

img = cv2.equalizeHist(img)

img_edges = cv2.Canny(img, 150, 150)

contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    
    area = cv2.contourArea(cnt)

    if area > 100:

        x, y, w, h = cv2.boundingRect(cnt)

        cv2.drawContours(img_copy, [cnt], -1, (0,255,0), 1)

        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255,0,0), 1)

        text_y = y - 5 if y - 5 > 10 else y + 15

        text = f"x:{x}, y:{y}, S:{int(area)}"
        cv2.putText(img_copy, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)


cv2.imshow("fake1", img)
cv2.imshow("original", img_copy)
cv2.imshow("fake2", img_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
