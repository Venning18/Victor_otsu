import numpy as np
from gray_hist import compute_gray_histogram

def otsu_threshold_float(p: np.ndarray) -> float:
    """
    Berechnet den Otsu-Schwellwert als Fließkommazahl (float), basierend auf der Histogrammverteilung p.
    """
    P = np.cumsum(p)
    bins = np.arange(len(p))
    mu = np.cumsum(bins * p)
    mu_T = mu[-1]

    sigma_b2 = (mu_T * P - mu)**2 / (P * (1 - P) + 1e-12)
    t_idx = np.argmax(sigma_b2)

    # Interpolation (optional, skimage-artig)
    if 0 < t_idx < len(sigma_b2) - 1:
        left, center, right = sigma_b2[t_idx - 1], sigma_b2[t_idx], sigma_b2[t_idx + 1]
        if right != left:
            offset = 0.5 * (right - left) / (right - 2 * center + left)
            return float(t_idx + offset)

    return float(t_idx)

def binarize(arr: np.ndarray, t: float) -> np.ndarray:
    return (arr > t).astype(np.uint8)

def apply_global_otsu(image: np.ndarray) -> np.ndarray:
    hist, _ = compute_gray_histogram(image)
    p = hist / hist.sum()
    t = otsu_threshold_float(p)
    return binarize(image, t)
