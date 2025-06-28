import numpy as np
from skimage.io import imread
from typing import Tuple, Union
from pathlib import Path

def load_image_and_gt(
    image_path: Union[str, Path],
    gt_path: Union[str, Path],
    threshold: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lädt ein Graustufenbild und die zugehörige Ground-Truth-Maske als bool-Arrays.

    Args:
        image_path: Pfad zum Eingabebild (Graustufenbild)
        gt_path: Pfad zur Ground Truth (Segmentierungsmaske)
        threshold: Schwellenwert für Binarisierung (z. B. 0.0 für alles > 0)

    Returns:
        image: Grauwertbild als np.ndarray (float, [0, 1])
        gt_mask: binäre Ground-Truth-Maske (dtype=bool)
    """
    image = imread(str(image_path), as_gray=True)
    gt_mask = imread(str(gt_path), as_gray=True) > threshold

    return image, gt_mask
