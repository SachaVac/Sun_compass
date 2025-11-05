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
    cx, cy = centroids[i]
    return (float(cx), float(cy)), mask

def uv_to_cam_dir(u, v, K, D):
    pts = np.array([[[u, v]]], dtype=np.float32)
    und = cv2.fisheye.undistortPoints(pts, K, D)
    x, y = und[0,0]

    v3 = np.array([x, y, 1.0])
    v3 /= np.linalg.norm(v3)
    return v3 

def cam_vec_to_local_az_el(v3):
    x, y, z = v3
    az = math.degrees(math.atan2(x, z))    
    el = math.degrees(math.asin(-y))      
    return az, el

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="calibration/see3cam_fisheye_calib.npz")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--percentile", type=float, default=99.0)
    args = parser.parse_args()


    try:
        SCRIPT_DIR = Path(__file__).resolve().parent
        import os; os.chdir(SCRIPT_DIR)
    except NameError:
        pass

    K, D, _ = load_calib(args.npz)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_AVFOUNDATION)  # macOS backend
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        raise SystemExit("ERROR1: no cam")

  

    t0=time.time(); frames=0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR2: cap.read() notworking."); break

        uv, mask = find_sun_uv(frame, hi_pct=args.percentile)
        print(uv,mask)
        overlay = frame.copy()
        status = "Sun: not found"

        if uv is not None:
            u, v = uv
            v3 = uv_to_cam_dir(u, v, K, D)
            print(v3)
            az, el = cam_vec_to_local_az_el(v3)
            print(az,el)
            
            cv2.drawMarker(overlay, (int(round(u)), int(round(v))), (0,0,255),
                           cv2.MARKER_TILTED_CROSS, 40, 2)


        cv2.putText(overlay, status, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        mh, mw = 240, 320
        m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if uv is not None else np.zeros((mh,mw,3), dtype=np.uint8)
        m3r = cv2.resize(m3, (mw, mh))
        overlay[0:mh, 0:mw] = m3r


        frames += 1
        if frames % 30 == 0:
            dt = time.time()-t0
            fps = frames/max(dt,1e-6)
            cv2.setWindowTitle("Sun detector", f"Sun detector – {fps:.1f} fps")

        cv2.imshow("Sun detector", overlay)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
