import numpy as np
from skimage.filters import threshold_otsu, threshold_local, threshold_multiotsu

from src.otsu_global import apply_global_otsu
from src.otsu_local import local_otsu
from src.load_image_pair import load_image_and_gt


def apply_skimage_global(image: np.ndarray) -> np.ndarray:
    """
    Globales Otsu aus skimage.filters.
    """
    t = threshold_otsu(image)
    return (image > t).astype(np.uint8)


def apply_skimage_local(image: np.ndarray, block_size: int = 35, offset: float = 0.0) -> np.ndarray:
    """
    Lokales Thresholding mit skimage.filters.threshold_local.
    """
    local_thresh = threshold_local(image, block_size, offset=offset)
    return (image > local_thresh).astype(np.uint8)


def apply_skimage_multiotsu(image: np.ndarray, classes: int = 2) -> np.ndarray:
    """
    Multi-Otsu-Schwellenwertverfahren von skimage.
    Für 2 Klassen ≈ einfaches Otsu, für 3+ auch z. B. Hintergrund / Zytoplasma / Zellkern.
    Gibt binäre Maske zurück: alles oberhalb erster Schwelle.
    """
    thresholds = threshold_multiotsu(image, classes=classes)
    return (image > thresholds[0]).astype(np.uint8)


def process_all_methods(image: np.ndarray) -> dict:
    """
    Wendet alle Segmentierungsmethoden auf das Bild an.

    Returns:
        Dictionary mit Methode → Binärbild (np.ndarray)
    """
    _, local_mask = local_otsu(image)

    return {
        "Otsu Global (custom)": apply_global_otsu(image),
        "Otsu Local (custom)": local_mask.astype(np.uint8),
        "Otsu Global (skimage)": apply_skimage_global(image),
        "Otsu Local (skimage)": apply_skimage_local(image),
        "Multi-Otsu (skimage)": apply_skimage_multiotsu(image)
    }
