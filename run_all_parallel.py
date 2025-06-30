
# run_all_parallel.py

import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from run_batch_evaluation import run_batch_evaluation
from visualize_segmentation import visualize_segmentations
from process_image import process_all_methods
from src.load_image_pair import load_image_and_gt

import matplotlib.pyplot as plt
import seaborn as sns

# ⚙️ Verzeichnisse definieren
base_data_dir = "data"
results_dir = "results"
visual_dir = "output_visuals"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(visual_dir, exist_ok=True)

# 📊 Segmentierung & Auswertung
all_dfs = []
print("📂 Starte Batch-Auswertung")
for dataset in os.listdir(base_data_dir):
    img_dir = os.path.join(base_data_dir, dataset, "img")
    gt_dir = os.path.join(base_data_dir, dataset, "gt")
    if not (os.path.isdir(img_dir) and os.path.isdir(gt_dir)):
        print(f"⚠️ Überspringe {dataset}, img/ oder gt/ fehlt.")
        continue

    print(f"🧪 Verarbeite Datensatz: {dataset}")
    try:
        df = run_batch_evaluation(img_dir, gt_dir, dataset=dataset)
        all_dfs.append(df)
    except Exception as e:
        print(f"❌ Fehler bei {dataset}: {e}")

# 📝 Ergebnisse speichern
if all_dfs:
    df_all = pd.concat(all_dfs, ignore_index=True)
    csv_path = os.path.join(results_dir, "dice_scores.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"✅ Ergebnisse gespeichert: {csv_path}")
else:
    print("⚠️ Keine Ergebnisse vorhanden.")
    exit()

# 🖼️ Parallele Visualisierung pro Bild
def process_image_task(args):
    dataset, file = args
    img_dir = os.path.join(base_data_dir, dataset, "img")
    gt_dir = os.path.join(base_data_dir, dataset, "gt")
    img_path = os.path.join(img_dir, file)

    if dataset == "NIH3T3":
        num = file.replace("dna-", "").replace(".png", "")
        gt_file = f"{num}.png"
    else:
        digits = ''.join(filter(str.isdigit, file))
        gt_file = f"man_seg{digits}.tif"
    gt_path = os.path.join(gt_dir, gt_file)
    if not os.path.exists(gt_path):
        return f"⚠️ GT fehlt für {file}, überspringe."

    try:
        image, gt_mask = load_image_and_gt(img_path, gt_path)
        predictions = process_all_methods(image)
        save_path = os.path.join(visual_dir, f"{dataset}_{file}.png")
        visualize_segmentations(image, gt_mask, predictions, save_path=save_path)
        return f"✅ Visualisiert: {file}"
    except Exception as e:
        return f"❌ Fehler bei Visualisierung von {file}: {e}"

print("🖼️ Starte parallele Bild-Visualisierung")
tasks = [
    (dataset, file)
    for dataset in os.listdir(base_data_dir)
    if os.path.isdir(os.path.join(base_data_dir, dataset, "img"))
    for file in os.listdir(os.path.join(base_data_dir, dataset, "img"))
    if file.endswith((".tif", ".png"))
]
with ProcessPoolExecutor() as executor:
    for msg in executor.map(process_image_task, tasks):
        print(msg)

# 📈 Vergleichsplots
print("📈 Erstelle Vergleichsplots")
df = pd.read_csv(os.path.join(results_dir, "dice_scores.csv"))

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Methode", y="Dice Score")
plt.xticks(rotation=45)
plt.title("Verteilung der Dice Scores je Methode")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "dice_scores_boxplot.png"), dpi=150)
plt.close()

# Heatmap
df_pivot = df.pivot(index="Bild", columns="Methode", values="Dice Score")
plt.figure(figsize=(12, 8))
sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5)
plt.title("Heatmap der Dice Scores")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "dice_scores_heatmap.png"), dpi=150)
plt.close()

# Scatterplots
def plot_scatter(df, methods, title, filename, color):
    df_sub = df[df["Methode"].isin(methods)]
    pivot = df_sub.pivot(index="Bild", columns="Methode", values="Dice Score")
    plt.figure(figsize=(6, 6))
    plt.scatter(pivot[methods[0]], pivot[methods[1]], color=color, s=60)
    plt.plot([0, 1], [0, 1], 'r--', label="Ideal: x = y")
    plt.xlabel(f"Dice Score – {methods[0]}")
    plt.ylabel(f"Dice Score – {methods[1]}")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=150)
    plt.close()

plot_scatter(df, ["Otsu Local (custom)", "Otsu Local (skimage)"],
             "Vergleich der Otsu Local Methoden", "otsu_local_scatterplot.png", "green")

plot_scatter(df, ["Otsu Global (custom)", "Otsu Global (skimage)"],
             "Vergleich der Otsu Global Methoden", "otsu_global_scatterplot.png", "blue")

print("🏁 Alle Schritte abgeschlossen.")
