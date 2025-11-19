#!/usr/bin/env python3
import cv2, numpy as np, math, argparse, time, os
from pathlib import Path

# --- Neměněné pomocné funkce ---

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
    cx, cy = centroids[i] # sun position on image in pixels
    return (float(cx), float(cy)), mask

def uv_to_cam_dir(u, v, K, D):
    pts = np.array([[[u, v]]], dtype=np.float32)
    # undistortPoints returns normalized image coordinates (x/z, y/z) when P is not provided.
    und = cv2.fisheye.undistortPoints(pts, K, D) # R and P are None by default
    x, y = und[0,0]
    v3 = np.array([x, y, 1.0]) # The 3D vector in camera coordinates (assuming z=1)
    v3 /= np.linalg.norm(v3) # Normalize to unit vector
    return v3 

def cam_vec_to_local_az_el(v3):
    x, y, z = v3
    az = math.degrees(math.atan2(x, z))    
    el = math.degrees(math.asin(-y))      
    return az, el

# --- Upravená hlavní funkce ---

def calculate_camsolar(calib_path, image_path_pattern, show_video=True):
    """
    Hledá polohu slunce v jednom nebo více snímcích.
    :param calib_path: Cesta k souboru s kalibrací (.npz).
    :param image_path_pattern: Vzor cesty pro načítání obrázků (např. '/data/img*.jpeg').
    :param show_video: Zda zobrazovat výsledky v okně.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=calib_path)
    parser.add_argument("--percentile", type=float, default=99.0)
    args, _ = parser.parse_known_args() 

    K, D, calib_size = load_calib(args.npz)
    if K is None:
        print("Could not load calibration file. Exiting.")
        return None, None, None

    calib_width, calib_height = calib_size
    image_files = list(Path(image_path_pattern.split('*')[0]).parent.glob(Path(image_path_pattern).name))
    
    if not image_files:
        print(f"ERROR: No images found matching pattern: {image_path_pattern}")
        return None, None, None
    
    print(f"Found {len(image_files)} images to process.")

    az, el, v3 = None, None, None
    
    # Procházení všech nalezených souborů
    for i, file_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {file_path}")
        
        # Načtení snímku z disku
        frame = cv2.imread(str(file_path))
        
        if frame is None:
            print(f"WARNING: Could not read image file: {file_path}. Skipping.")
            continue
            
        # Zkontrolujeme a změníme velikost, pokud neodpovídá kalibraci
        if frame.shape[1] != calib_width or frame.shape[0] != calib_height:
             print(f"WARNING: Image size {frame.shape[1]}x{frame.shape[0]} does not match calibration size {calib_width}x{calib_height}. Resizing image.")
             frame = cv2.resize(frame, (calib_width, calib_height))


        uv, mask = find_sun_uv(frame, hi_pct=args.percentile)

        if uv is not None:
            u, v = uv
            v3 = uv_to_cam_dir(u, v, K, D)
            az, el = cam_vec_to_local_az_el(v3)
            print(f"Sun found in image {file_path}: Az: {az:.1f}, El: {el:.1f}")
            
            # Pokud nepotřebujeme zobrazovat video, můžeme skončit po nalezení slunce v prvním obrázku
            if not show_video:
                return az, el, v3

        if show_video:
            overlay = frame.copy()
            if uv is not None:
                u, v = uv
                status = f"Sun found | Az: {az:.1f} El: {el:.1f} | {file_path.name}"
                cv2.drawMarker(overlay, (int(round(u)), int(round(v))), (0,0,255),
                               cv2.MARKER_TILTED_CROSS, 40, 2)
            else:
                status = f"Sun: not found | {file_path.name}"

            cv2.putText(overlay, status, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
            mh, mw = 240, 320
            if mask is not None:
                m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                #m3r = cv2.resize(m3, (mw, mh))
                #overlay[0:mh, 0:mw] = m3r
                cv2.imshow("Mask", m3)

            cv2.imshow("Sun detector", overlay)
            # Čeká na stisknutí klávesy po dobu 1 milisekundy (pro video efekt)
            # Pokud chceme, aby se zobrazil každý obrázek dokud stiskneme klávesu: cv2.waitKey(0)
            k = cv2.waitKey(1) & 0xFF 
            input()
            if k in (27, ord('q')): break

    cv2.destroyAllWindows()
    print("Processing finished.")
    # Vrátí výsledky z posledního zpracovaného snímku
    return az, el, v3

if __name__ == '__main__':
    # Příklad použití pro obrázky. Změňte cesty podle potřeby!
    calib_path = "../Vision/calibration/see3cam_fisheye_calib.npz"
    # Nastavte cestu k adresáři a vzor pro soubory, např. /data/img001.jpeg, /data/img002.jpeg, ...
    # Zde předpokládáme, že existuje adresář 'data' ve stejném adresáři jako skript
    # a uvnitř jsou soubory s názvem začínajícím 'img' a končícím '.jpeg'
    image_pattern = "data/IMG*.jpeg" 
    
    # Upozornění: Funkce se pokusí zpracovat VŠECHNY odpovídající obrázky. 
    # V režimu show_video (True) se bude zobrazovat okno pro každý snímek.
    # Pokud chcete pouze spočítat výsledek z prvního obrázku, který nalezne Slunce, nastavte show_video=False
    az_cam, el_cam, s_cam = calculate_camsolar(calib_path, image_pattern, show_video=True)
    
    if az_cam is not None:
        print("\n--- Results (Last Processed Image) ---")
        print(f"Camera Azimuth: {az_cam:.2f} degrees")
        print(f"Camera Elevation: {el_cam:.2f} degrees")
        print(f"Camera Sun Vector (s_r): {s_cam}")