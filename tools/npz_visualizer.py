import numpy as np
import pandas as pd

npz_path = "see3cam_fisheye_calib.npz"

# Načtení dat
data = np.load(npz_path)

print("--- Dostupné klíče ---")
print(list(data.keys()))
print("-----------------------")

# Projdeme všechny uložené prvky a zobrazíme je
for key in data.keys():
    arr = data[key]
    print(f"\n🔑 Klíč: {key}, Tvar: {arr.shape}, Typ: {arr.dtype}")
    
    # Zobrazení matic K a D v tabulkovém formátu
    if len(arr.shape) == 2:
        # Použijeme Pandas pro hezčí tabulkové zobrazení
        df = pd.DataFrame(arr)
        print(df)
    else:
        # Zobrazení vektorů nebo skalárů
        print(arr)

# POZNÁMKA: V Jupyter Notebooku se pole K nebo D zobrazí 
# jako interaktivní tabulka, když napíšete např.:
# data['K']