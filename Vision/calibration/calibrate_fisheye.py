import cv2, numpy as np, glob
import os

# Set working dir
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("working_dir", os.getcwd())

CHECKERBOARD = (8, 5)
SQUARE = 0.03
images = sorted(glob.glob("calib_images/calib_*.jpg"))

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= SQUARE

objpoints, imgpoints = [], []
img_shape = None
cntr = 0

for f in images:
    cntr+=1
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
    if ret:
        if img_shape is None:
            img_shape = gray.shape[::-1]
        corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        objpoints.append(objp)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('calib', img)
        cv2.waitKey(50)
cv2.destroyAllWindows()
print(cntr)
if not objpoints:
    raise SystemExit("Nothing found")

K = np.zeros((3,3))
D = np.zeros((4,1))
rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints, imgpoints, img_shape, K, D, None, None,
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
)
print("RMS:", rms)
print("K=\n", K)
print("D=\n", D)
np.savez("see3cam_fisheye_calib.npz", K=K, D=D, size=np.array(img_shape))
