import os
from glob import glob
import matplotlib.pyplot as plt
from src.load_image_pair import load_image_and_gt
from process_image import process_all_methods

# Basisordner
base_data_dir = "data"
output_base = "output_visuals"
os.makedirs(output_base, exist_ok=True)

# Alle Datensätze in 'data/' durchgehen
for dataset_name in os.listdir(base_data_dir):
    dataset_path = os.path.join(base_data_dir, dataset_name)
    img_dir = os.path.join(dataset_path, "img")
    gt_dir  = os.path.join(dataset_path, "gt")

    if not (os.path.isdir(img_dir) and os.path.isdir(gt_dir)):
        print(f"⚠️  Überspringe {dataset_name}: img oder gt fehlt.")
        continue

    print(f"\n📂 Verarbeite Visualisierung für {dataset_name}")
    os.makedirs(os.path.join(output_base, dataset_name), exist_ok=True)

    # Unterstützte Bildformate
    img_paths = []
    for ext in ["*.tif", "*.png"]:
        img_paths.extend(glob(os.path.join(img_dir, ext)))
    img_paths.sort()

    for img_path in img_paths:
        basename = os.path.basename(img_path)

        if dataset_name == "NIH3T3":
            num = basename.replace("dna-", "").replace(".png", "")
            gt_filename = f"{num}.png"
        else:
            num = ''.join(filter(str.isdigit, basename))
            gt_filename = f"man_seg{num}.tif"

        gt_path = os.path.join(gt_dir, gt_filename)
        if not os.path.exists(gt_path):
            print(f"⚠️  GT fehlt für {basename}, überspringe.")
            continue

        try:
            # Bild & GT laden
            image, gt_mask = load_image_and_gt(img_path, gt_path)
            predictions = process_all_methods(image)

            # Visualisierung
            for method_name, mask in predictions.items():
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(image, cmap="gray")
                axes[0].set_title("Originalbild")
                axes[1].imshow(gt_mask, cmap="gray")
                axes[1].set_title("Ground Truth")
                axes[2].imshow(mask, cmap="gray")
                axes[2].set_title(f"Segmentierung: {method_name}")

                for ax in axes:
                    ax.axis("off")

                out_dir = os.path.join(output_base, dataset_name, method_name)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, basename.replace(".tif", ".png"))
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close()

                print(f"✅ Gespeichert: {out_path}")

        except Exception as e:
            print(f"❌ Fehler bei {basename}: {e}")
