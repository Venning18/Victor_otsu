
# Zellkernsegmentierung mit Otsu-Verfahren

Dieses Projekt implementiert verschiedene Otsu-basierte Segmentierungsmethoden und vergleicht deren Leistung anhand des Dice Scores mit Ground-Truth-Daten. Dazu gehören sowohl eigene Implementierungen als auch Referenzverfahren aus `scikit-image`.

---

## Voraussetzungen

### Installation der Abhängigkeiten

Installiere die benötigten Pakete mit:

```bash
pip install -r requirements.txt
```

Falls du lieber ein Python-Skript nutzt:

```bash
python install_requirements.py
```

### `requirements.txt` enthält:

- `numpy` – numerische Operationen
- `pandas` – Datenverarbeitung (z. B. für Dice Scores)
- `matplotlib` – Visualisierung von Bildern & Scores
- `seaborn` – erweiterte Plot-Darstellung (Boxplots, Heatmaps)
- `scikit-image` – Bildverarbeitung und Segmentierung
- `tqdm` – Fortschrittsbalken für Batch-Auswertungen

---

## Projektstruktur

```plaintext
.
├── data/                    # Mikroskopiebilder & Ground Truth
├── src/                    # Segmentierungsmodule (Otsu, Histogramm, Dice Score)
├── output_visuals/         # Visualisierte Segmentierungsergebnisse
├── results/                # CSV & PNG-Auswertungen (z. B. Dice Scores)
├── requirements.txt        # Alle benötigten Python-Pakete
├── install_requirements.py # Alternativer Installer (Python)
├── run_batch_evaluation.py # Führt Segmentierung & Bewertung auf ganzen Datensätzen durch
├── process_image.py        # Enthält alle Methoden zur Segmentierung
├── evaluate_segmentation.py# Berechnet Dice Scores
├── visualize_segmentations.py # Darstellung der Segmentierungsergebnisse
└── README.md               # Projektbeschreibung
```

---

## Erste Schritte

1. **Daten in `data/` ablegen**
2. **`run_batch_evaluation.py` ausführen**, um Segmentierungen und Dice Scores zu erzeugen.
3. **Ergebnisse mit `plot_all_methods.py` visualisieren**.

---

## Ziel

Vergleich und Bewertung unterschiedlicher Segmentierungsmethoden zur Zellkernextraktion aus Mikroskopieaufnahmen.

---

## Kontakt & Weiterentwicklung

Feedback, Erweiterungen oder Anfragen gerne als Pull Request oder direkt per Nachricht.
