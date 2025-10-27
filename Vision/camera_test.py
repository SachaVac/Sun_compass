import cv2

cap = cv2.VideoCapture(0)  # macOS: 0 bývá defaultní kamera
if not cap.isOpened():
    print("Kamera se neotevřela.")
    exit()

print("Rozlišení:", cap.get(cv2.CAP_PROP_FRAME_WIDTH, ), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv2.CAP_PROP_FPS, 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("See3CAM test", frame)
    if cv2.waitKey(1) == 27:  # ESC pro ukončení
        break

cap.release()
cv2.destroyAllWindows()
