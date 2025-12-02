import cv2, numpy as np, math, argparse, time, os

parser = argparse.ArgumentParser()
parser.add_argument("--npz", default="calibration/see3cam_fisheye_calib.npz")
parser.add_argument("--cam", type=int, default=0)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1200)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--percentile", type=float, default=99.0)
args = parser.parse_args()




cap = cv2.VideoCapture(args.cam, cv2.CAP_AVFOUNDATION)  # macOS backend
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
cap.set(cv2.CAP_PROP_FPS,          args.fps)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

if not cap.isOpened():
    raise SystemExit("ERROR1: no cam")

# Vypnout auto-expozici (na Linuxu UVC = 1 = manual, 3 = auto)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

# Nastavit manuální expozici – příklad: 500 μs (hodnota závisí na kameře)
# Některé kamery používají záporné hodnoty, např. -5, -8, atd.
cap.set(cv2.CAP_PROP_EXPOSURE, 10)

while True:
    ret, frame = cap.read()
    cv2.imshow("camera", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
