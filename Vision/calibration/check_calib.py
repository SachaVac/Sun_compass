import cv2, numpy as np
import os
# Set working dir
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("working_dir", os.getcwd())


data = np.load("see3cam_fisheye_calib.npz")
K, D = data["K"], data["D"]
img = cv2.imread("calib_images/calib_024.jpg")
h, w = img.shape[:2]
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w,h), cv2.CV_16SC2)
undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
cv2.imshow("original", img)
cv2.imshow("undistorted", undist)
cv2.waitKey(0)
