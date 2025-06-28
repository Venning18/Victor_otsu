from run_batch_evaluation import run_batch_evaluation
import os
import pandas as pd

if __name__ == "__main__":
    base_data_dir = "data"
    all_dfs = []

    for dataset_name in os.listdir(base_data_dir):
        dataset_path = os.path.join(base_data_dir, dataset_name)
        img_dir = os.path.join(dataset_path, "img")
        gt_dir  = os.path.join(dataset_path, "gt")

        if not (os.path.isdir(img_dir) and os.path.isdir(gt_dir)):
            print(f"⚠️  Überspringe {dataset_name}, 'img/' oder 'gt/' fehlt.")
            continue

        print(f"📂 Verarbeite Datensatz: {dataset_name}")
        try:
            df = run_batch_evaluation(img_dir, gt_dir, dataset=dataset_name)
            all_dfs.append(df)
        except Exception as e:
            print(f"   ❌ Fehler bei {dataset_name}: {e}")

    # Gesamt-Datenframe erstellen
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        os.makedirs("results", exist_ok=True)
        df_all.to_csv("results/dice_scores.csv", index=False)
        print(f"\n✅ Ergebnisse gespeichert: results/dice_scores.csv")
        print(df_all)
    else:
        print("⚠️ Keine Ergebnisse gesammelt.")
 