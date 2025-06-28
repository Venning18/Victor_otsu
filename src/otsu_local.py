import numpy as np
from skimage import img_as_ubyte
from src.gray_hist import compute_gray_histogram
from src.otsu_global import otsu_threshold

def local_otsu(image: np.ndarray, radius: int = 3) -> tuple[np.ndarray, np.ndarray]:
    img_u8 = img_as_ubyte(image)
    H, W = img_u8.shape
    t_map = np.zeros((H, W), dtype=np.uint8)
    mask  = np.zeros((H, W), dtype=bool)

    pad = radius
    padded = np.pad(img_u8, pad, mode="reflect")
    w = 2 * radius + 1

    for i in range(H):
        for j in range(W):
            block = padded[i : i + w, j : j + w]
            hist, _ = compute_gray_histogram(block)
            p = hist / hist.sum()
            t = otsu_threshold(p)
            t_map[i, j] = t
            mask[i, j] = (img_u8[i, j] > t)

    return t_map, mask
