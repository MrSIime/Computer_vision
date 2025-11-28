import cv2 # type: ignore
import numpy as np # type: ignore
from numpy.ma.core import filled # type: ignore

img = np.zeros((512, 512, 3), np.uint8)

# img[:] = 100, 250, 120

cv2.rectangle(img, (100, 100), (250, 350), (0, 255, 0), -1)
cv2.line(img, (0, 0), (512, 512), (255, 0, 0), 5)
cv2.line(img, ((img.shape[1] // 2), (img.shape[0] // 2)), (256, 512), (255, 200, 0), 5)
cv2.circle(img, (400, 100), 50, (0, 0, 255), -1)
cv2.putText(img, "OpenCV", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)    

print(img.shape)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()