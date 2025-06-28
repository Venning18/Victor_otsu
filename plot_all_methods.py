import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pfad zur CSV-Datei
csv_path = os.path.join("results", "dice_scores.csv")

# Sicherstellen, dass die Datei existiert
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")

# CSV laden
df = pd.read_csv(csv_path)

# 🔹 BOXPLOT: Verteilung der Dice Scores je Methode
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Methode", y="Dice Score")
plt.xticks(rotation=45)
plt.title("Verteilung der Dice Scores je Methode")
plt.tight_layout()
plt.savefig("results/dice_scores_boxplot.png", dpi=150)
plt.show()

# 🔹 HEATMAP: Dice Scores pro Bild & Methode
df_pivot = df.pivot(index="Bild", columns="Methode", values="Dice Score")

plt.figure(figsize=(12, 8))
sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5)
plt.title("Heatmap der Dice Scores (Bilder × Methoden)")
plt.tight_layout()
plt.savefig("results/dice_scores_heatmap.png", dpi=150)
plt.show()
