import cv2 # type: ignore
import numpy as np # type: ignore
from numpy.ma.core import filled # type: ignore

img = cv2.imread("images/notRacist.jpg")

s = img.shape

cv2.rectangle(img, ((2*(s[1] // 5)), (s[0] // 5)), (4*(s[1] // 5), 5*(s[0] // 6)), (0, 0, 255), 5)
cv2.putText(img, "not racist", ((2*(s[1] // 5)), (s[0] // 14)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 2)    
cv2.putText(img, "happy", ((2*(s[1] // 5)), (s[0] // 14)+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 2)   
cv2.putText(img, "age: 45-50", ((2*(s[1] // 5)), (s[0] // 14)+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 2)   


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()