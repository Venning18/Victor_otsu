import numpy as np
from skimage.io import imread

def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Berechnet den Dice-Koeffizienten zwischen zwei binären Bildern (dtype=bool).
    """
    if pred.shape != target.shape:
        raise ValueError("Die Eingabebilder haben unterschiedliche Formen.")

    intersection = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()

    if total == 0:
        return 1.0  # Sonderfall: beide leer → perfekte Übereinstimmung

    return 2 * intersection / total

if __name__ == "__main__":
    pred = imread("data-git/N2DH-GOWT1/img/t01.tif", as_gray=True) > 0
    gt   = imread("data-git/N2DH-GOWT1/gt/man_seg01.tif", as_gray=True) > 0
    print(f"Dice Score: {dice_score(pred, gt):.4f}")
