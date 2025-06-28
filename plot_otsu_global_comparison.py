import os
import pandas as pd
import matplotlib.pyplot as plt

# Pfad zur Ergebnisdatei (ggf. anpassen)
csv_path = os.path.join("results", "dice_scores.csv")

# Prüfen ob Datei existiert
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")

# CSV einlesen
df = pd.read_csv(csv_path)

# Filter auf die beiden gewünschten Methoden
df_global = df[df["Methode"].isin(["Otsu Global (custom)", "Otsu Global (skimage)"])]

# Pivotieren → Zeilen = Bild, Spalten = Methode
df_pivot = df_global.pivot(index="Bild", columns="Methode", values="Dice Score")

# Scatterplot erzeugen
plt.figure(figsize=(6, 6))
plt.scatter(
    df_pivot["Otsu Global (custom)"],
    df_pivot["Otsu Global (skimage)"],
    color="blue", s=60
)
plt.plot([0, 1], [0, 1], 'r--', label="Ideal: x = y")
plt.xlabel("Dice Score – Otsu Global (eigener Code)")
plt.ylabel("Dice Score – Otsu Global (skimage)")
plt.title("Vergleich der Otsu Global Methoden")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Optional: Speichern
plt.savefig("results/otsu_global_scatterplot.png", dpi=150)
plt.show()
