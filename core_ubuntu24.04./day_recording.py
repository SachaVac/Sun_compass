import cv2
import time
import os
from datetime import datetime

# --- KONFIGURACE ---
CAM_INDEX = 2
WIDTH, HEIGHT = 1920, 1200
TARGET_FPS = 6  # Nativní FPS kamery pro UYVY mode
OUTPUT_DIR = "zaznamy_final"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def record_stable():
    print(f"[{datetime.now()}] Inicializace See3CAM v režimu UYVY (Low Bandwidth)...")
    
    # Použijeme V4L2 backend
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    
    # NASTAVENÍ HARDWARU NA 6 FPS (UYVY)
    # Toto zajistí, že kamera neposílá zbytečná data
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'UYVY'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # Ověření nastavení
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # Převod fourcc na čitelný text
    codec_text = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"Kamera nastavena na: {codec_text} @ {cap.get(cv2.CAP_PROP_FPS)} FPS")

    # VideoWriter pro AI (MP4V je spolehlivý)
    file_path = os.path.join(OUTPUT_DIR, f"zaznam_{datetime.now().strftime('%H-%M-%S')}.mp4")
    out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), TARGET_FPS, (WIDTH, HEIGHT))

    print("Nahrávání běží. Pro ukončení stiskněte Ctrl+C.")

    try:
        while cap.isOpened():
            # Čtení snímku (teď jich chodí jen 6 za sekundu, takže žádný spěch)
            ret, frame = cap.read()
            
            if not ret:
                # Pokud nastane timeout, zkusíme krátce počkat a číst znovu
                print("Chvilkový výpadek dat, čekám...")
                time.sleep(0.1)
                continue

            # Vložení časové značky pro AI segmentaci
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            cv2.putText(frame, timestamp, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            out.write(frame)
            
            # Žádné time.sleep() zde není potřeba, cap.read() se sám synchronizuje s kamerou

    except KeyboardInterrupt:
        print("\nZáznam ukončen uživatelem.")
    finally:
        cap.release()
        out.release()
        print(f"Soubor uložen: {file_path}")

if __name__ == "__main__":
    record_stable()