import subprocess
import os
import time
from datetime import datetime

# --- KONFIGURACE ---
CAM_INDEX = "/dev/video2"
WIDTH, HEIGHT = 1920, 1200
FPS = 5  # Požadované FPS v MP4
OUTPUT_DIR = "zaznamy_final_mp4"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def record_with_ffmpeg():
    print(f"[{datetime.now()}] Spouštím robustní nahrávání přes FFmpeg...")
    
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = os.path.join(OUTPUT_DIR, f"zaznam_{timestamp}.mp4")
        
        # FFmpeg příkaz:
        # -f v4l2: použije ovladač Linuxu
        # -input_format uyvy: použije stabilní 6fps režim kamery
        # -i: vstupní zařízení
        # -r: vynutí 5 FPS na výstupu
        # -c:v libx264: nejlepší komprese pro MP4
        # -preset ultrafast: minimální zátěž CPU
        # -crf 25: rozumný poměr kvalita/velikost (vyšší číslo = menší soubor)
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'v4l2',
            '-input_format', 'uyvy',
            '-video_size', f'{WIDTH}x{HEIGHT}',
            '-i', CAM_INDEX,
            '-r', str(FPS),
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '25',
            '-an', # bez zvuku
            file_path
        ]
        
        print(f"[{datetime.now()}] Nahrávám do: {file_path}")
        
        try:
            # Spustí nahrávání a čeká, dokud neskončí (nebo nespadne)
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            
            # Sledujeme stderr, abychom viděli případné chyby v reálném čase
            while process.poll() is None:
                time.sleep(1)
            
            _, stderr = process.communicate()
            print(f"[{datetime.now()}] FFmpeg se zastavil. Důvod: {stderr}")
            
        except Exception as e:
            print(f"Chyba při běhu FFmpeg: {e}")
        
        print("Pokus o restart nahrávání za 2 sekundy...")
        time.sleep(2)

if __name__ == "__main__":
    try:
        record_with_ffmpeg()
    except KeyboardInterrupt:
        print("\nMonitorování ukončeno uživatelem.")