import numpy as np
from src.gray_hist import compute_gray_histogram

def otsu_threshold(p: np.ndarray) -> int:
    P = np.cumsum(p)
    bins = np.arange(len(p))
    mu = np.cumsum(bins * p)
    mu_T = mu[-1]
    sigma_b2 = (mu_T * P - mu)**2 / (P * (1 - P) + 1e-12)
    return int(np.argmax(sigma_b2))

def binarize(arr: np.ndarray, t: int) -> np.ndarray:
    return (arr > t).astype(np.uint8)

def apply_global_otsu(image: np.ndarray) -> np.ndarray:
    hist, _ = compute_gray_histogram(image)
    p = hist / hist.sum()
    t = otsu_threshold(p)
    return binarize(image, t)
