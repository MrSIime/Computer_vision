import cv2 # type: ignore
import numpy as np # type: ignore

image = cv2.imread("images/1.jpg")

resize = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
resize = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

image = cv2.Canny(resize, 100, 100)

kernel = np.ones((5,5), np.uint8)
image = cv2.dilate(image, kernel, iterations=1)
image = cv2.erode(image, kernel, iterations=1)
cv2.imwrite("Image.jpg", image)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()