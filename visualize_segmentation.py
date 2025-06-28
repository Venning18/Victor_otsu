import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

def visualize_segmentations(
    image: np.ndarray,
    gt_mask: np.ndarray,
    predictions: Dict[str, np.ndarray],
    max_cols: int = 3,
    save_path: str = None
):
    """
    Zeigt Originalbild, Ground Truth und Segmentierungsergebnisse nebeneinander.

    Args:
        image: Originalbild (Graustufen, float [0,1])
        gt_mask: Ground Truth Maske (bool)
        predictions: Dict {Methodenname: Binärmaske (0/1 oder bool)}
        max_cols: maximale Anzahl an Spalten pro Zeile
        save_path: optionaler Pfad zum Abspeichern der Abbildung als PNG
    """
    all_items = [("Original", image), ("Ground Truth", gt_mask)] + list(predictions.items())
    n = len(all_items)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flat if n > 1 else [axes]

    for ax, (title, img) in zip(axes, all_items):
        cmap = "gray" if img.ndim == 2 else None
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[len(all_items):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Visualisierung gespeichert unter: {save_path}")
    plt.show()
