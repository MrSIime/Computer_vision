import cv2

# img = cv2.imread("images/1.jpg")
# img = cv2.resize(img, (500, 300))

# cv2.imshow("img", img)

cap = cv2.VideoCapture("video/1.mp4")

while True:
    ret, frame = cap.read()

    resize = cv2.resize(frame, (500, 200))

    cv2.imshow("video", resize)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if not ret:
        break

cap.release()
# cv2.waitKey(0)
cv2.destroyAllWindows()