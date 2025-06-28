import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Union, Tuple

def compute_gray_histogram(
    image_source: Union[Path, str, np.ndarray],
    bins: int = 256,
    value_range: Tuple[int, int] = (0, 255)
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(image_source, (Path, str)):
        img = Image.open(str(image_source)).convert("L")
        arr = np.array(img)
    elif isinstance(image_source, np.ndarray):
        arr = image_source
    else:
        raise TypeError("Erwartet Pfad oder NumPy-Array.")

    hist, bin_edges = np.histogram(arr.ravel(), bins=bins, range=value_range)
    return hist, bin_edges

def plot_gray_histogram(hist: np.ndarray, bin_edges: np.ndarray):
    plt.figure(figsize=(8, 4))
    plt.bar(bin_edges[:-1], hist, width=bin_edges[1] - bin_edges[0], align='edge')
    plt.xlabel("Grauwert")
    plt.ylabel("Häufigkeit")
    plt.title("Grauwert-Histogramm")
    plt.show()
