import pandas as pd
from src.dice_score import dice_score
from typing import Dict
import numpy as np

def evaluate_segmentations(
    gt_mask: np.ndarray,
    predictions: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Berechnet die Dice Scores aller Segmentierungsmethoden.

    Args:
        gt_mask: Ground Truth Maske (bool)
        predictions: Dict {Methodenname: Binärmaske (0/1 oder bool)}

    Returns:
        DataFrame mit Methode und zugehörigem Dice Score
    """
    results = []

    for name, pred in predictions.items():
        dice = dice_score(pred.astype(bool), gt_mask.astype(bool))
        results.append({"Methode": name, "Dice Score": dice})

    df = pd.DataFrame(results).sort_values("Dice Score", ascending=False)
    return df
