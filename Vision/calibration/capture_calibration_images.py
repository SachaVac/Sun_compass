import cv2, time, os

# parametry
cam_index = 0
out_dir = "calib_images_new"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise SystemExit("❌ Kamera se neotevřela.")

cv2.namedWindow("See3CAM_24CUG", cv2.WINDOW_NORMAL)
cv2.resizeWindow("See3CAM_24CUG", 960, 600)

counter = len(os.listdir(out_dir))
print(f"[INFO] Spuštěno – stiskni 's' pro uložení snímku, 'q' pro konec")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] žádný snímek"); break

    text = f"Počet: {counter}  |  's' uložit  |  'q' konec"
    cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("See3CAM_24CUG", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        fname = os.path.join(out_dir, f"calib_{counter:03d}.jpg")
        cv2.imwrite(fname, frame)
        print(f"[SAVED] {fname}")
        counter += 1
        time.sleep(0.3)  # malá pauza proti dvojkliku
    elif k in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Hotovo.")
