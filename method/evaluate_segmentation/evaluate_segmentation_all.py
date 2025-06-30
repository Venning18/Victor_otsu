import os
from glob import glob
from src.load_image_pair import load_image_and_gt
from process_image import process_all_methods
from method.evaluate_segmentation.evaluate_segmentation import evaluate_segmentations

# Basisverzeichnis für alle Datensätze
base_data_dir = "data"

# Gehe durch alle Unterordner von 'data'
for dataset_name in os.listdir(base_data_dir):
    dataset_path = os.path.join(base_data_dir, dataset_name)
    img_dir = os.path.join(dataset_path, "img")
    gt_dir  = os.path.join(dataset_path, "gt")

    if not (os.path.isdir(img_dir) and os.path.isdir(gt_dir)):
        print(f"⚠️  Überspringe {dataset_name}, 'img/' oder 'gt/' fehlt.")
        continue

    print(f"\n📂 Datensatz: {dataset_name}")

    # Unterstützte Bildtypen
    image_extensions = ["*.tif", "*.png"]
    img_paths = []
    for ext in image_extensions:
        img_paths.extend(glob(os.path.join(img_dir, ext)))

    img_paths.sort()

    for img_path in img_paths:
        basename = os.path.basename(img_path)

        # 🔁 Fallunterscheidung je nach Datensatz
        if dataset_name == "NIH3T3":
            # z. B. dna-0.png → 0.png
            num = basename.replace("dna-", "").replace(".png", "")
            gt_filename = f"{num}.png"
        else:
            # z. B. t01.tif → man_seg01.tif
            num = ''.join(filter(str.isdigit, basename))
            gt_filename = f"man_seg{num}.tif"

        gt_path = os.path.join(gt_dir, gt_filename)

        if not os.path.exists(gt_path):
            print(f"   ⚠️  GT fehlt für {basename} → erwartet: {gt_filename}")
            continue

        try:
            # Bild & GT laden
            image, gt_mask = load_image_and_gt(img_path, gt_path)

            # Segmentierung durchführen
            results = process_all_methods(image)

            # Dice Scores berechnen
            df_scores = evaluate_segmentations(gt_mask, results)

            print(f"\n   📊 Dice Scores für {basename}:")
            print(df_scores)

        except Exception as e:
            print(f"   ❌ Fehler bei {basename}: {e}")
 