import os
import pandas as pd
from glob import glob
from tqdm import tqdm

from src.load_image_pair import load_image_and_gt
from process_image import process_all_methods
from method.evaluate_segmentation.evaluate_segmentation import evaluate_segmentations


def run_batch_evaluation(img_dir, gt_dir, dataset=None):
    """
    Führt die Segmentierung und Auswertung für alle Bild-GT-Paare durch.

    Args:
        img_dir: Verzeichnis mit Input-Bildern
        gt_dir: Verzeichnis mit Ground-Truth-Bildern
        dataset: Optionaler Datensatzname (für Logging / Export)

    Returns:
        DataFrame mit allen Dice Scores (Bild × Methode)
    """
    records = []

    # Alle vorhandenen GT-Dateinamen (ohne Endung) merken
    gt_names = set(os.path.splitext(f)[0] for f in os.listdir(gt_dir))

    # Alle Bilder erfassen
    img_paths = []
    for ext in ["*.tif", "*.png"]:
        img_paths.extend(glob(os.path.join(img_dir, ext)))
    img_paths.sort()

    for img_path in tqdm(img_paths, desc="Verarbeite Bilder"):
        basename = os.path.basename(img_path)

        # Zuordnung der GT-Datei
        if dataset == "NIH3T3":
            num = basename.replace("dna-", "").replace(".png", "")
            gt_filename = f"{num}.png"
        else:
            num = ''.join(filter(str.isdigit, basename))
            gt_filename = f"man_seg{num}.tif"

        gt_path = os.path.join(gt_dir, gt_filename)

        if not os.path.exists(gt_path):
            print(f"⚠️  Ground Truth fehlt für {basename}, überspringe.")
            continue

        try:
            # Lade Bild & GT
            image, gt_mask = load_image_and_gt(img_path, gt_path)

            # Segmentierungen berechnen
            predictions = process_all_methods(image)

            # Dice Scores berechnen
            scores = evaluate_segmentations(gt_mask, predictions)

            # Metadaten ergänzen
            scores["Bild"] = basename
            if dataset:
                scores["Datensatz"] = dataset

            records.append(scores)

        except Exception as e:
            print(f"❌ Fehler bei {basename}: {e}")
            continue

    if not records:
        print(f"⚠️  Keine gültigen Bild-GT-Paare in {img_dir}")
        return pd.DataFrame(columns=["Bild", "Methode", "Dice Score", "Datensatz"])

    # Alle Scores zusammenführen
    df_all = pd.concat(records, ignore_index=True)
    return df_all[["Bild", "Methode", "Dice Score", "Datensatz"] if dataset else ["Bild", "Methode", "Dice Score"]]
