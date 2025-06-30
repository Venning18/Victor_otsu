import numpy as np
from gray_hist import compute_gray_histogram

def otsu_threshold_float(p: np.ndarray, bin_edges: np.ndarray) -> float:
    """
    Berechnet den Otsu-Schwellwert als Fließkommazahl (float), basierend auf der Histogrammverteilung p
    und den zugehörigen Bin-Grenzen.
    """
    P = np.cumsum(p)
    bins = np.arange(len(p))
    mu = np.cumsum(bins * p)
    mu_T = mu[-1]

    sigma_b2 = (mu_T * P - mu)**2 / (P * (1 - P) + 1e-12)
    t_idx = np.argmax(sigma_b2)

    # Interpolation zwischen Bin-Kanten
    if 0 < t_idx < len(sigma_b2) - 1:
        left, center, right = sigma_b2[t_idx - 1], sigma_b2[t_idx], sigma_b2[t_idx + 1]
        if right != left:
            offset = 0.5 * (right - left) / (right - 2 * center + left)
            t_float = bin_edges[0] + (t_idx + offset) * (bin_edges[1] - bin_edges[0])
            return float(t_float)

    # Kein Offset → direkt aus Bin-Mitte
    t_float = bin_edges[0] + t_idx * (bin_edges[1] - bin_edges[0])
    return float(t_float)

def binarize(arr: np.ndarray, t: float) -> np.ndarray:
    return (arr > t).astype(np.uint8)

def apply_global_otsu(image: np.ndarray) -> np.ndarray:
    hist, bin_edges = compute_gray_histogram(image)
    p = hist / hist.sum()
    t = otsu_threshold_float(p, bin_edges)
    return binarize(image, t)
