import os
from glob import glob
from src.load_image_pair import load_image_and_gt
from process_image import process_all_methods

# Hauptdatenverzeichnis
base_data_dir = "data"

# Unterstützte Bildformate
image_extensions = ["*.tif", "*.png"]

# Alle Datensätze in data/
for dataset_name in os.listdir(base_data_dir):
    dataset_path = os.path.join(base_data_dir, dataset_name)
    img_dir = os.path.join(dataset_path, "img")
    gt_dir  = os.path.join(dataset_path, "gt")

    if not (os.path.isdir(img_dir) or not os.path.isdir(gt_dir)):
        print(f"⚠️  Überspringe {dataset_name}: img/ oder gt/ fehlt.")
        continue

    print(f"\n📂 Verarbeite Datensatz: {dataset_name}")

    # Bilder erfassen
    img_paths = []
    for ext in image_extensions:
        img_paths.extend(glob(os.path.join(img_dir, ext)))
    img_paths.sort()

    for img_path in img_paths:
        basename = os.path.basename(img_path)

        # Bildnummer für GT ableiten
        if dataset_name == "NIH3T3":
            num = basename.replace("dna-", "").replace(".png", "")
            gt_filename = f"{num}.png"
        else:
            num = ''.join(filter(str.isdigit, basename))
            gt_filename = f"man_seg{num}.tif"

        gt_path = os.path.join(gt_dir, gt_filename)

        if not os.path.exists(gt_path):
            print(f"   ⚠️  GT fehlt für {basename}, überspringe.")
            continue

        try:
            image, gt = load_image_and_gt(img_path, gt_path)
            results = process_all_methods(image)

            print(f"\n🖼️  {basename} ({image.shape}):")
            for name, mask in results.items():
                print(f"   🔹 {name}: {mask.shape}, Positiv: {mask.sum()}")

        except Exception as e:
            print(f"   ❌ Fehler bei {basename}: {e}")
