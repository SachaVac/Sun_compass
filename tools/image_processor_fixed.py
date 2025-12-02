#!/usr/bin/env python3
import cv2
import numpy as np
import math
import argparse
import time
import os
from pathlib import Path

# --- pomocné funkce ---
def load_calib(npz_path):
    data = np.load(npz_path)
    K, D = data["K"], data["D"]
    size = tuple(int(x) for x in data.get("size", [1920, 1200]))
    return K, D, size

def find_sun_uv(frame, hi_pct=99.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lin = np.clip((gray / 255.0) ** 2.2, 0, 1)
    thr = np.percentile(lin, hi_pct)
    mask = (lin >= thr).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num < 2:
        return None, mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = 1 + int(np.argmax(areas))
    cx, cy = centroids[i]
    return (float(cx), float(cy)), mask

# --- numerická inverze theta_d -> theta ---
def invert_fisheye_theta(r_d, D, tol=1e-8, max_iter=20):
    """Newton-Raphson inverze OpenCV fisheye: r_d -> theta"""
    k1, k2, k3, k4 = D.flatten()
    theta = r_d  # initial guess
    for _ in range(max_iter):
        theta2 = theta**2
        theta4 = theta2**2
        theta6 = theta2 * theta4
        theta8 = theta4**2
        f = theta * (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8) - r_d
        df = 1 + 3*k1*theta2 + 5*k2*theta4 + 7*k3*theta6 + 9*k4*theta8
        theta_new = theta - f/df
        if abs(theta_new - theta) < tol:
            break
        theta = theta_new
    return theta

def uv_to_cam_dir(u, v, K, D):
    """Pixel -> přesný 3D jednotkový vektor v kamerových souřadnicích"""
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy
    y_n = -y_n  # y nahoru

    r_d = math.sqrt(x_n**2 + y_n**2)
    if r_d < 1e-12:
        theta = 0.0
    else:
        theta = invert_fisheye_theta(r_d, D)
    if r_d > 1e-12:
        vx = math.sin(theta) * x_n / r_d
        vy = math.sin(theta) * y_n / r_d
    else:
        vx, vy = 0.0, 0.0
    vz = math.cos(theta)
    v3 = np.array([vx, vy, vz])
    v3 /= np.linalg.norm(v3)
    return v3

def cam_vec_to_local_az_el(v3):
    """3D vektor -> azimuth 0–360°, elevation 0–90°"""
    x, y, z = v3
    az = math.degrees(math.atan2(x, z))
    if az < 0:
        az += 360
    el = math.degrees(math.asin(y))
    return az, el

# --- hlavní funkce ---
def calculate_camsolar(npz_path, cam_index=0, show_video=True):
    K, D, (width, height) = load_calib(npz_path)

    cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)  # macOS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("Kamera nelze otevřít")

    az, el, v3 = None, None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        uv, mask = find_sun_uv(frame)
        if uv is not None:
            u, v = uv
            v3 = uv_to_cam_dir(u, v, K, D)
            az, el = cam_vec_to_local_az_el(v3)
            print(f"Sun found: Az={az:.1f}°, El={el:.1f}°")

        if show_video:
            overlay = frame.copy()
            if uv is not None:
                cv2.drawMarker(overlay, (int(round(u)), int(round(v))), (0,0,255),
                               cv2.MARKER_TILTED_CROSS, 40, 2)
                status = f"Az={az:.1f}° El={el:.1f}°"
            else:
                status = "Sun not found"
            cv2.putText(overlay, status, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow("Sun detector", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    cap.release()
    cv2.destroyAllWindows()
    return az, el, v3

# --- spustitelná část ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="../Vision/calibration/see3cam_fisheye_calib.npz")
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()

    az, el, v3 = calculate_camsolar(args.npz, cam_index=args.cam, show_video=True)
    if az is not None:
        print("\n--- Results ---")
        print(f"Camera Azimuth: {az:.2f}°")
        print(f"Camera Elevation: {el:.2f}°")
        print(f"Camera Sun Vector: {v3}")
