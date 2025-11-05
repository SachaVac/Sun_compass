#!/usr/bin/env python3
import cv2, numpy as np, math, argparse, time, os
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("working_dir", os.getcwd())

def load_calib(npz_path="see3cam_fisheye_calib.npz"):
    data = np.load(npz_path)
    K, D = data["K"], data["D"]
    size = tuple(int(x) for x in data.get("size", [1920,1200]))
    #breakpoint()
    return K, D, size

def find_sun_uv(frame, hi_pct=99.0):
    # top percentil of brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lin = np.clip((gray/255.0)**2.2, 0, 1)
    thr = np.percentile(lin, hi_pct)
    mask = (lin >= thr).astype(np.uint8)*255
    mask = cv2.medianBlur(mask, 5)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num < 2:
        return None, mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = 1 + int(np.argmax(areas))
    cx, cy = centroids[i] # sun position on image in pixels
    return (float(cx), float(cy)), mask

def uv_to_cam_dir(u, v, K, D):
    pts = np.array([[[u, v]]], dtype=np.float32)
    # undistortPoints returns normalized image coordinates (x/z, y/z) when P is not provided.
    und = cv2.fisheye.undistortPoints(pts, K, D) # R and P are None by default
    #breakpoint()
    x, y = und[0,0]
    v3 = np.array([x, y, 1.0]) # The 3D vector in camera coordinates (assuming z=1)
    v3 /= np.linalg.norm(v3) # Normalize to unit vector
    return v3 

def cam_vec_to_local_az_el(v3):
    x, y, z = v3
    az = math.degrees(math.atan2(x, z))    
    el = math.degrees(math.asin(-y))      
    return az, el

def calculate_camsolar(path, show_video=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=path)
    parser.add_argument("--cam", type=int, default=0)    
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--percentile", type=float, default=99.0)
    args, _ = parser.parse_known_args() # Use parse_known_args to avoid conflicts


    try:
        SCRIPT_DIR = Path(__file__).resolve().parent
        import os; os.chdir(SCRIPT_DIR)
    except NameError:
        pass

    K, D, calib_size = load_calib(args.npz)
    if K is None:
        print("Could not load calibration file. Exiting.")
        return None, None, None

    calib_width, calib_height = calib_size

    cap = cv2.VideoCapture(args.cam, cv2.CAP_AVFOUNDATION)  # macOS backend
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  calib_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib_height)
    cap.set(cv2.CAP_PROP_FPS,          args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    # Verify that the camera is using the resolution from calibration
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_width != calib_width or actual_height != calib_height:
        raise SystemExit(f"ERROR: Camera is running at {actual_width}x{actual_height}, but calibration is for {calib_width}x{calib_height}.")

    if not cap.isOpened():
        raise SystemExit("ERROR1: no cam")

  

    t0=time.time()
    # Try for a few seconds to find the sun
    while time.time() - t0 < 5.0:
        ok, frame = cap.read()
        if not ok:
            print("ERROR2: cap.read() not working."); 
            time.sleep(0.1)
            continue

        uv, mask = find_sun_uv(frame, hi_pct=args.percentile)

        if uv is not None:
            #breakpoint()
            u, v = uv
            v3 = uv_to_cam_dir(u, v, K, D)
            az, el = cam_vec_to_local_az_el(v3)
            print(f"Sun found in camera frame: Az: {az:.1f}, El: {el:.1f}")
            
            if not show_video:
                cap.release()
                cv2.destroyAllWindows()
                return az, el, v3

        if show_video:
            overlay = frame.copy()
            if uv is not None:
                u, v = uv
                status = f"Sun found | Az: {az:.1f} El: {el:.1f}"
                cv2.drawMarker(overlay, (int(round(u)), int(round(v))), (0,0,255),
                               cv2.MARKER_TILTED_CROSS, 40, 2)
            else:
                status = "Sun: not found"

            cv2.putText(overlay, status, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
            mh, mw = 240, 320
            if mask is not None:
                m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                m3r = cv2.resize(m3, (mw, mh))
                overlay[0:mh, 0:mw] = m3r

            cv2.imshow("Sun detector", overlay)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')): break
    
    cap.release()
    cv2.destroyAllWindows()
    print("5s done")
    return az, el, v3

if __name__ == '__main__':
    # Example of how to use the function
    # Path to your calibration file
    calib_path = "../Vision/calibration/see3cam_fisheye_calib.npz"
    az_cam, el_cam, s_cam = calculate_camsolar(calib_path, show_video=True)
    if az_cam is not None:
        print("\n--- Results ---")
        print(f"Camera Azimuth: {az_cam:.2f} degrees")
        print(f"Camera Elevation: {el_cam:.2f} degrees")
        print(f"Camera Sun Vector (s_r): {s_cam}")
