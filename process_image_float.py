import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

# Projektstruktur anpassen
project_root = os.getcwd()
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Eigene Module importieren
from otsu_global import compute_gray_histogram, otsu_threshold, binarize
from otsu_global_float import otsu_threshold_float

# Bildpfad
img_path = os.path.join(project_root, "Data", "N2DL-HeLa", "img", "t13.tif")

# Bild laden (Graustufen)
img = Image.open(img_path).convert("L")
img_array = np.array(img)

# Histogramm & Wahrscheinlichkeiten berechnen
hist, bin_edges = compute_gray_histogram(img_array)
p = hist / np.sum(hist)

# 1. Otsu Global (eigene Implementierung, integer)
t_own = otsu_threshold(p)
mask_own = binarize(img_array, t_own)

# 2. Otsu Global Float (eigene float-Implementierung mit bin_edges)
t_float = otsu_threshold_float(p, bin_edges)
mask_float = binarize(img_array, t_float)

# 3. Otsu aus skimage
t_sk = threshold_otsu(img_array)
mask_sk = binarize(img_array, t_sk)

# Schwellenwerte ausgeben
print(f"Otsu Threshold (eigener Code): t = {t_own}")
print(f"Otsu Threshold (float): t = {t_float:.2f}")
print(f"Otsu Threshold (skimage): t = {t_sk}")

# Vergleichende Visualisierung
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
axs[0].imshow(img_array, cmap="gray")
axs[0].set_title("Original")

axs[1].imshow(mask_own, cmap="gray")
axs[1].set_title(f"Eigener Otsu (t={t_own})")

axs[2].imshow(mask_float, cmap="gray")
axs[2].set_title(f"Otsu Float (t={t_float:.2f})")

axs[3].imshow(mask_sk, cmap="gray")
axs[3].set_title(f"Skimage Otsu (t={t_sk})")

for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.show()

# Debugging-Ausgaben
print("Bildwerte:", img_array.dtype, img_array.min(), img_array.max())
print("Histogramm Summe:", np.sum(hist))
print("Histogramm Max:", np.max(hist))
print("p sum:", np.sum(p), " min/max:", np.min(p), np.max(p))
