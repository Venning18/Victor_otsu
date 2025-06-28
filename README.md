# 🧬 Zellbild-Segmentierung mit Otsu-Verfahren

Dieses Projekt vergleicht verschiedene Methoden zur automatischen Segmentierung von Zellkernen in fluoreszenzmikroskopischen Aufnahmen. Der Fokus liegt auf der Implementierung und dem Vergleich von globalen und lokalen Otsu-Schwellenwertverfahren.

---

## 🎯 Ziel

Ziel des Projekts ist es, die Segmentierungsqualität eigener Otsu-Implementierungen (global und lokal) gegenüber etablierten Methoden (`skimage`) zu bewerten. Als Referenz dienen handsegmentierte Ground-Truth-Masken.

---

## ⚙️ Methoden

Folgende Methoden wurden implementiert und verglichen:

| Methode                        | Quelle       | Beschreibung                            |
|-------------------------------|--------------|-----------------------------------------|
| Otsu Global (custom)          | Eigene       | Globales Otsu-Verfahren via Histogramm  |
| Otsu Global (skimage)         | `skimage`    | Referenz-Implementierung                |
| Otsu Local (custom)           | Eigene       | Lokaler Schwellenwert pro Pixel         |
| Otsu Local (skimage)          | `skimage`    | Lokales Mittelwert-Verfahren            |
| Multi-Otsu (skimage)          | `skimage`    | Schwellenwerte für 3 Klassen            |

Die Segmentierungen werden mit dem **Dice Score** gegen die Ground-Truth bewertet.

---

## 📊 Ergebnisse

Die Bewertung erfolgt quantitativ (Dice Score) und visuell:

- **Boxplots** zur Verteilung der Scores pro Methode
- **Heatmap** der Scores pro Bild × Methode
- **Scatterplots**: eigene vs. `skimage`-Methode
- **Visualisierungen** der Segmentierungen mit Original- und GT-Bild

Beispielhafte Ergebnisse finden sich im Ordner `output_visuals/`.

---

## 📁 Projektstruktur

```
.
├── data/                      # Eingabebilder & Ground-Truth
├── results/                   # Dice Scores (CSV)
├── output_visuals/           # Segmentierungs-Bilder
├── src/
│   ├── otsu_global.py        # Eigene globale Otsu-Methode
│   ├── otsu_local.py         # Eigene lokale Otsu-Methode
│   ├── gray_hist.py          # Histogramm-Funktionen
│   ├── load_image_pair.py    # Bild & GT laden
├── process_image.py          # Wendet alle Methoden auf ein Bild an
├── run_batch_evaluation.py   # Wendet alle Methoden auf alle Bilder an
├── visualize_segmentation.py # Speichert Beispielbilder
├── Otsu_Segmentierung_Auswertung.ipynb  # Ergebnisnotebook
└── README.md
```

---

## ▶️ Ausführung

1. Python 3.8+ installieren  
2. Abhängigkeiten installieren (z. B. via `requirements.txt`)

```bash
pip install -r requirements.txt
```

3. Segmentierung ausführen:

```bash
python run_all_batch_evaluation.py
```

4. Visualisierung:

```bash
python visualize_segmentation.py
```

5. Ergebnisse im Notebook auswerten:

```bash
jupyter notebook Otsu_Segmentierung_Auswertung.ipynb
```

---

## 🧪 Beispielausgabe

<img src="output_visuals/N2DH-GOWT1/Otsu Global (custom)/t01.png" width="500" />

---

## 👤 Autor

Victor De Souza Enning – Molekulare Biotechnologie, Universität Heidelberg

---

## 📜 Lizenz

MIT License – frei verwendbar mit Nennung
