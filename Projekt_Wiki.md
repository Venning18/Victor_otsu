
# 📂 Projekt-Wiki: Otsu-basierte Zellbildsegmentierung

Dieses Wiki dokumentiert die Struktur, Funktion und Anwendung eines Projekts zur Segmentierung von Zellkernen in Mikroskopiebildern mithilfe globaler und lokaler Otsu-Verfahren sowie Referenzmethoden aus skimage. 

---

## 🔺 Projektstruktur

```
.
├── data/                      # Eingabebilder & Ground-Truths (manuelle Segmentierungen)
│   ├── N2DH-GOWT1/
│   │   ├── img/               # Rohbilder (z. B. "t01.tif")
│   │   └── gt/                # Manuelle Segmentierungen ("man_seg01.tif")
│   ├── N2DL-HeLa/             # Weiterer Zellbild-Datensatz
│   │   ├── img/
│   │   └── gt/
│   └── NIH3T3/                # Zelllinien-Datensatz ("dna-0.png" mit "0.png" als GT)
│       ├── img/
│       └── gt/
├── output_visuals/           # Visualisierte Segmentierungen (Original + GT + Ergebnis)
├── results/                  # CSV-Dateien mit Dice Scores pro Methode
├── src/                      # Quellcode-Module (eigene Methoden)
│   ├── dice_score.py
│   ├── gray_hist.py
│   ├── load_image_pair.py
│   ├── otsu_global.py
│   └── otsu_local.py
├── process_image.py          # Kombiniert alle Methoden für ein Bild
├── run_batch_evaluation.py   # Bewertet automatisch ganze Bildersätze
├── run_all_evaluations.py    # Führt Bewertung für alle Datensätze aus
├── visualize_segmentation.py # Speichert Visualisierungen als PNG
├── plot_dice_score_summary.py # Heatmap- & Boxplot-Analyse der Dice-Scores
├── plot_otsu_scatter.py      # Scatterplots für Global Otsu (custom vs skimage)
├── Otsu_Segmentierung_Auswertung.ipynb  # Jupyter Notebook zur Ergebnisanalyse
└── README.md                 # Projektbeschreibung
```


---

## 🧠 Zielsetzung

Ziel des Projekts ist es, Zellkerne aus Fluoreszenz-Mikroskopieaufnahmen automatisch zu segmentieren. Dabei werden verschiedene Varianten des Otsu-Schwellenwertverfahrens eingesetzt und quantitativ mit der Ground-Truth-Segmentierung verglichen.

---

## 🔧 Modulübersicht

## 🧮 `dice_score.py` – Berechnung des Dice-Koeffizienten

### Zweck

Dieses Skript berechnet den **Dice-Koeffizienten**, eine gängige Metrik zur Bewertung der Überlappung zwischen zwei binären Segmentierungen (z. B. Vorhersage vs. Ground Truth). Der Dice Score liegt zwischen 0 (keine Überlappung) und 1 (perfekte Übereinstimmung).

---

### 📦 Importe

```python
import numpy as np
from skimage.io import imread
```

- `numpy` wird für numerische Operationen auf Arrays verwendet.
- `skimage.io.imread` lädt Bilddateien als `NumPy`-Arrays.

---

### 🔧 Funktion `dice_score(pred, target)`

```python
def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
```

#### Argumente:
- `pred`: vorhergesagte binäre Maske (z. B. von einem Segmentierungsalgorithmus), Typ: `np.ndarray`, dtype: `bool`
- `target`: Ground-Truth-Maske (manuell annotiert), ebenfalls ein binäres `np.ndarray`

#### Ablauf:
1. **Formprüfung**:
   ```python
   if pred.shape != target.shape:
       raise ValueError("Die Eingabebilder haben unterschiedliche Formen.")
   ```
   Beide Arrays müssen dieselbe Form haben (Pixelanzahl in Höhe und Breite identisch), sonst wird ein Fehler geworfen.

2. **Berechnung der Überlappung (Intersection)**:
   ```python
   intersection = np.logical_and(pred, target).sum()
   ```
   Dies zählt die Anzahl an Pixeln, bei denen **beide Masken gleichzeitig `True`** sind.

3. **Berechnung der Gesamtanzahl positiver Pixel**:
   ```python
   total = pred.sum() + target.sum()
   ```
   Dies zählt **alle positiven Pixel** in beiden Masken.

4. **Sonderfallbehandlung**:
   ```python
   if total == 0:
       return 1.0
   ```
   Wenn **beide Masken komplett leer** sind (kein einziger positiver Pixel), ist das Ergebnis perfekt (Score = 1).

5. **Berechnung des Dice Scores**:
   ```python
   return 2 * intersection / total
   ```
   Dies ist die Formel für den Dice-Koeffizienten:
   \[
   \text{Dice} = \frac{2 \cdot |A \cap B|}{|A| + |B|}
   \]

---

### 🧪 Testlauf im `__main__`-Block

```python
if __name__ == "__main__":
    pred = imread("data-git/N2DH-GOWT1/img/t01.tif", as_gray=True) > 0
    gt   = imread("data-git/N2DH-GOWT1/gt/man_seg01.tif", as_gray=True) > 0
    print(f"Dice Score: {dice_score(pred, gt):.4f}")
```

#### Erklärung:
- `imread(..., as_gray=True)` lädt das Bild als Graustufenbild.
- `> 0` binarisiert das Bild (alle Pixel > 0 werden zu `True`, alle anderen zu `False`).
- Danach wird `dice_score(pred, gt)` berechnet und mit 4 Dezimalstellen ausgegeben.

#### Beispielausgabe:
```
Dice Score: 0.8234
```

Dies bedeutet: Es besteht eine Überlappung von etwa 82,34 % zwischen vorhergesagter und tatsächlicher Segmentierung.

---

### ✅ Zusammenfassung

- Diese Funktion ist **zentral für die quantitative Bewertung** von Segmentierungsergebnissen.
- Sie ist robust gegenüber leeren Masken.
- Sie kann leicht in größere Pipelines eingebunden werden, um **viele Segmentierungsmethoden vergleichbar zu machen**.


# 🧮 `gray_hist.py` – Grauwert-Histogrammberechnung

Dieses Modul dient der Berechnung und optionalen Visualisierung von Histogrammen für Graustufenbilder. Es wird u.a. für die Otsu-Segmentierung verwendet, um die Intensitätsverteilung eines Bildes auszuwerten.

---

## 🔧 Funktionen

### 1. `compute_gray_histogram(image_source, bins=256, value_range=(0, 255))`

#### 📌 Zweck
Diese Funktion berechnet das Histogramm eines Grauwertbildes.

#### 📥 Eingaben:
- `image_source` (`Path`, `str` oder `np.ndarray`): Das Bild kann entweder als Pfad oder direkt als NumPy-Array übergeben werden.
- `bins` (`int`): Anzahl der Bins im Histogramm. Standard: 256 (für 8-Bit-Bilder sinnvoll).
- `value_range` (`Tuple[int, int]`): Wertebereich der Intensitäten. Standardmäßig von 0 bis 255.

#### 📤 Ausgabe:
- `hist`: Array mit den Häufigkeiten der Grauwerte.
- `bin_edges`: Array mit den Bin-Grenzen.

#### 🔍 Funktionsweise:
1. Bild wird geladen (falls ein Pfad übergeben wurde).
2. In ein Grauwertbild konvertiert.
3. Flach gemacht mit `.ravel()`, damit `np.histogram` funktioniert.
4. Histogramm mit `np.histogram()` berechnet.

---

### 2. `plot_gray_histogram(hist, bin_edges)`

#### 📌 Zweck
Visualisiert ein Histogramm mit `matplotlib`.

#### 📥 Eingaben:
- `hist`: Die berechneten Häufigkeiten (z. B. aus `compute_gray_histogram`).
- `bin_edges`: Die zugehörigen Intensitätsgrenzen.

#### 🔍 Funktionsweise:
- Erstellt ein Balkendiagramm (`plt.bar`).
- Achsenbeschriftungen: "Grauwert" und "Häufigkeit".
- Anzeige mit `plt.show()`.

---

## 🧪 Beispielanwendung

```python
from gray_hist import compute_gray_histogram, plot_gray_histogram
hist, bin_edges = compute_gray_histogram("path/to/image.png")
plot_gray_histogram(hist, bin_edges)
```

---

## 📎 Hinweise
- Diese Funktionen sind wichtig für Schwellenwertverfahren wie das Otsu-Verfahren.
- Sie erlauben eine Analyse der Bildhelligkeit und helfen bei der automatischen Segmentierung.



# 📄 Modul: `load_image_pair.py`

Dieses Skript enthält eine zentrale Hilfsfunktion zum Laden von Bild-Ground-Truth-Paaren aus einem gegebenen Dateipfad. Es wird verwendet, um sowohl das Eingabebild (z. B. ein Graustufen-Mikroskopiebild) als auch die zugehörige Ground-Truth-Segmentierungsmaske korrekt zu laden und als `NumPy`-Arrays zurückzugeben.

---

## 🧠 Ziel

Die Funktion stellt sicher, dass:
- Bilder korrekt als Grauwertbilder gelesen werden,
- die Ground-Truth-Maske korrekt binarisiert wird (also in ein boolesches Format überführt wird),
- die Rückgabe für weitere Segmentierungs- und Evaluierungsschritte geeignet ist.

---

## 🧩 Code-Übersicht

```python
import numpy as np
from skimage.io import imread
from typing import Tuple, Union
from pathlib import Path
```

### 📌 Funktion: `load_image_and_gt(...)`

```python
def load_image_and_gt(
    image_path: Union[str, Path],
    gt_path: Union[str, Path],
    threshold: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
```

### 🔍 Parameter:

- `image_path`: Pfad zum Bild (String oder Pathlib-Objekt)
- `gt_path`: Pfad zur Ground-Truth-Maske
- `threshold`: Schwellenwert, um die GT-Maske in eine binäre Maske umzuwandeln (Standard: 0.0)

### 🔄 Rückgabe:

Ein Tupel bestehend aus:
- `image`: Graustufenbild als 2D `np.ndarray` mit Werten im Bereich [0,1]
- `gt_mask`: Binäre Ground-Truth-Maske (dtype=bool)

---

## 🔧 Funktionsweise im Detail:

```python
image = imread(str(image_path), as_gray=True)
```
- Das Bild wird mithilfe von `skimage.io.imread` als Graustufenbild geladen (automatisch normalisiert auf Bereich [0,1]).

```python
gt_mask = imread(str(gt_path), as_gray=True) > threshold
```
- Die Ground-Truth-Maske wird ebenfalls als Grauwertbild geladen.
- Durch den Vergleich `> threshold` wird das Bild in eine binäre Maske umgewandelt, z. B. alles was größer als 0 ist, wird als „True“ interpretiert.

---

## ✅ Beispiel:

```python
image, gt = load_image_and_gt("data/N2DH-GOWT1/img/t01.tif", "data/N2DH-GOWT1/gt/man_seg01.tif")
```

- Lädt das Bild `t01.tif` und die zugehörige Ground-Truth-Maske `man_seg01.tif` aus dem Dataset `N2DH-GOWT1`.

---

## 📌 Hinweise

- Der Schwellenwert kann angepasst werden, falls GT-Bilder in Grauwerten vorliegen.
- Diese Funktion stellt sicher, dass sowohl Bild als auch Maske kompatibel weiterverarbeitet werden können.


# 🧠 otsu_global.py

Dieses Modul enthält die eigene Implementierung des globalen Otsu-Verfahrens zur Schwellenwertbestimmung und Segmentierung von Grauwertbildern.

---

## 📌 Funktionen

### `otsu_threshold(p: np.ndarray) -> int`
Berechnet den optimalen Schwellenwert `t` anhand einer Wahrscheinlichkeitsverteilung `p` (normalisiertes Histogramm).

**Ablauf:**
1. `P = np.cumsum(p)` → Kumulative Summe der Wahrscheinlichkeiten.
2. `bins = np.arange(len(p))` → Grauwertachsen.
3. `mu = np.cumsum(bins * p)` → Kumulative Mittelwerte.
4. `mu_T = mu[-1]` → Gesamtmittelwert.
5. `sigma_b2 = (mu_T * P - mu)**2 / (P * (1 - P) + 1e-12)` → Between-class variance.
6. `argmax(...)` → Der optimale Schwellenwert maximiert diese Varianz.

### `binarize(arr: np.ndarray, t: int) -> np.ndarray`
Binarisiert ein Grauwertbild: Alle Pixel größer als `t` werden als 1 (Objekt) gesetzt.

**Rückgabe:** Binärbild (0 = Hintergrund, 1 = Objekt).

### `apply_global_otsu(image: np.ndarray) -> np.ndarray`
Volle Pipeline:
1. Histogramm berechnen mit `compute_gray_histogram(...)`
2. Histogramm normalisieren zu Wahrscheinlichkeiten
3. Schwellenwert mit `otsu_threshold(...)` berechnen
4. Binärbild erzeugen mit `binarize(...)`

---

## 🧪 Beispielverwendung

```python
from otsu_global import apply_global_otsu
from skimage.io import imread
image = imread("path/to/image.tif", as_gray=True)
binary = apply_global_otsu(image)
```

---

## 🔗 Abhängigkeiten

- `gray_hist.py`: für die Histogrammberechnung
- `numpy`: für mathematische Operationen



# 📄 `otsu_local.py`

Dieses Modul implementiert eine lokale Version des Otsu-Schwellenwertverfahrens. Statt eines einzigen globalen Schwellenwerts wird für jeden Pixel ein individueller Schwellenwert berechnet, basierend auf seinem lokalen Umfeld.

## 🔧 Funktionen

### `local_otsu(image: np.ndarray, radius: int = 3) -> tuple[np.ndarray, np.ndarray]`

Diese Funktion berechnet für jedes Pixel eines Grauwertbildes einen lokalen Otsu-Schwellenwert und segmentiert das Bild auf dieser Basis.

#### Parameter:
- `image` (`np.ndarray`): Eingabebild im Wertebereich [0, 1].
- `radius` (`int`, Standard = 3): Die Umgebung jedes Pixels wird als Quadrat mit Seitenlänge `2 * radius + 1` betrachtet.

#### Rückgabewerte:
- `t_map` (`np.ndarray`): Schwellenwertkarte (gleiche Größe wie das Eingabebild).
- `mask` (`np.ndarray`): Binäre Maske (True = Objekt, False = Hintergrund).

## 🧠 Funktionsweise

1. **Vorverarbeitung:** Das Eingabebild wird mit `skimage.img_as_ubyte` in ein 8-bit Bild konvertiert.
2. **Padding:** Um Ränder zu berücksichtigen, wird das Bild mit `np.pad` erweitert.
3. **Lokale Verarbeitung:** Für jedes Pixel wird ein Block (Patch) um den Pixel extrahiert:
   - Es wird das Grauwert-Histogramm des Blocks berechnet.
   - Daraus wird mittels Otsu (aus `otsu_global.otsu_threshold`) ein lokaler Schwellenwert bestimmt.
   - Der aktuelle Pixelwert wird mit diesem Schwellenwert verglichen.
4. **Ausgabe:** Die resultierende Schwellenwertkarte und Segmentierungsmaske werden zurückgegeben.

## 📦 Abhängigkeiten

- `numpy`
- `skimage.img_as_ubyte`
- `src.gray_hist.compute_gray_histogram`
- `src.otsu_global.otsu_threshold`

---

Diese Methode ist besonders nützlich bei Bildern mit inhomogener Ausleuchtung oder variierenden Kontrasten, da sie kontextabhängig segmentiert.


# 🧪 Modulbeschreibung: `process_image.py`, `process_image_ein.py`, `process_image_all.py`

Dieses Modul-Set bündelt alle Schritte zur Anwendung und Auswertung verschiedener Bildsegmentierungsmethoden auf einzelne oder mehrere Mikroskopie-Bilder. Im Fokus stehen Otsu-basierte Verfahren sowie Referenzimplementierungen aus `skimage`.

---

## 🔧 `process_image.py`

### Funktion `process_all_methods(image)`

Diese Funktion nimmt ein einzelnes Grauwertbild entgegen und wendet folgende Segmentierungsmethoden darauf an:

- **Otsu Global (custom)**: Eigene Implementierung basierend auf Histogramm-Analyse.
- **Otsu Local (custom)**: Berechnet für jedes Pixel einen lokalen Schwellenwert basierend auf Histogrammen im Umkreis.
- **Otsu Global (skimage)**: Standard-Implementierung aus `skimage.filters.threshold_otsu`.
- **Otsu Local (skimage)**: Lokale Schwellenwertmethode (`threshold_local`) mit anpassbarem Fenster.
- **Multi-Otsu (skimage)**: Erweiterung für mehr als zwei Klassen – nützlich zur Trennung von Zellkern, Zytoplasma und Hintergrund.

**Rückgabewert:** Ein Dictionary `{Methodenname: Binärmaske (np.ndarray)}`

### Helferfunktionen:

- `apply_skimage_global(...)` – verwendet globalen Otsu aus `skimage`
- `apply_skimage_local(...)` – verwendet lokalen Schwellenwert aus `skimage`
- `apply_skimage_multiotsu(...)` – verwendet `threshold_multiotsu` für mehrklassige Segmentierung

---

## 🧪 `process_image_ein.py`

### Ziel:

- Testweise Anwendung der Methoden auf genau ein Bild.
- Nutzt: `load_image_and_gt(...)` um Bild und Ground Truth zu laden.
- Führt `process_all_methods(image)` aus.
- Gibt **Form** und **Anzahl positiver Pixel** jeder Methode in der Konsole aus.

### Beispielausgabe:

```
Otsu Global (custom): (512, 512), Positiv: 12034
Otsu Local (custom):  (512, 512), Positiv: 13400
...
```

---

## 🔁 `process_image_all.py`

### Ziel:

- Läuft automatisiert über alle Datensätze im Verzeichnis `data/`.
- Erkennt automatisch, ob ein Bild zur **NIH3T3**-Serie gehört (andere Benennung).
- Lädt jedes Bild mit zugehöriger Ground Truth.
- Führt `process_all_methods(image)` aus.
- Gibt pro Bild und Methode die Dimension und Anzahl der Segmentierungspixel in der Konsole aus.

### Hinweise zur Funktion:

- Unterstützt `.tif` und `.png`
- Erkennt fehlende Ground-Truth-Dateien automatisch und überspringt diese.
- Gut geeignet für erste visuelle Kontrolle der Pipeline auf kompletten Datensätzen.

---

## 📌 Zusammenfassung

Diese drei Dateien bilden gemeinsam das Rückgrat der Bildverarbeitungspipeline:

| Datei                 | Aufgabe                                      |
|----------------------|----------------------------------------------|
| `process_image.py`   | Definiert alle Segmentierungsmethoden        |
| `process_image_ein.py` | Einzelbild-Analyse (Debugging, Test)       |
| `process_image_all.py` | Batch-Anwendung auf komplette Datensätze   |

# 📊 evaluate_segmentation.py – Auswertung der Segmentierung mittels Dice Score

Dieses Modul bewertet die Qualität verschiedener Segmentierungsmethoden, indem es den sogenannten **Dice Score** verwendet. Der Dice-Koeffizient ist ein gängiges Maß zur Überlappung binärer Masken in der Bildverarbeitung.

---

## 🧠 Funktionsweise

### Funktion: `evaluate_segmentations(gt_mask, predictions)`

Diese Funktion berechnet den Dice Score für jede Segmentierungsmethode im Vergleich zur Ground-Truth-Maske.

#### Argumente:
- `gt_mask` (`np.ndarray`): Die Ground-Truth-Maske (boolesches Array)
- `predictions` (`Dict[str, np.ndarray]`): Ein Dictionary mit Namen der Methoden und den binären Segmentierungsmasken

#### Rückgabe:
- `pd.DataFrame`: Eine sortierte Tabelle mit den Dice Scores aller Methoden

### Rechenprinzip

Für jede Methode wird die Vorhersagemaske in ein boolesches Format umgewandelt. Dann wird mithilfe der Funktion `dice_score` die Übereinstimmung zur Ground Truth berechnet.

---

## 🧪 Beispiel: Anwendung auf ein einzelnes Bild

```python
from src.load_image_pair import load_image_and_gt
from process_image import process_all_methods
from evaluate_segmentation import evaluate_segmentations

image, gt_mask = load_image_and_gt("data/N2DH-GOWT1/img/t01.tif",
                                   "data/N2DH-GOWT1/gt/man_seg01.tif")

results = process_all_methods(image)
df_scores = evaluate_segmentations(gt_mask, results)
print(df_scores)
```

---

## 📁 Anwendung auf ganze Datensätze

Ein weiterer Abschnitt des Codes iteriert durch alle Bilder eines Datensatzes. Dabei wird für jedes Bild die Segmentierung berechnet, mit der Ground Truth verglichen und die Dice Scores ausgegeben.

#### Ablauf:
1. Iteration über alle Ordner in `data/`
2. Laden aller Bilder aus `img/`
3. Zuordnung der passenden Ground-Truth-Datei aus `gt/`
4. Aufruf von:
    - `load_image_and_gt`
    - `process_all_methods`
    - `evaluate_segmentations`
5. Ausgabe der Dice Scores pro Bild und Methode

Dieser automatisierte Ablauf dient zur vergleichenden Evaluation über ganze Datensätze hinweg.

---

## ⚠️ Fehlerbehandlung

- Bilder ohne passende Ground Truth werden übersprungen
- Fehler beim Laden oder Verarbeiten eines Bildes werden abgefangen und ausgegeben

---

## ✅ Nutzen

Dieses Skript ist essenziell für die **quantitative Bewertung** der Segmentierungsmethoden. Es zeigt zuverlässig, welche Methode in welchem Datensatz die besten Ergebnisse liefert.
---

## 📊 Batch-Auswertung


# 📄 Dokumentation: `run_batch_evaluation.py`

Dieses Skript automatisiert die Auswertung von Segmentierungsmethoden durch Vergleich der segmentierten Masken mit Ground-Truth-Daten. Es verwendet den Dice-Koeffizienten zur Bewertung der Genauigkeit.

---

## 🔧 Funktionen

### `run_batch_evaluation(img_dir, gt_dir, dataset=None)`

- **Zweck:** Führt Segmentierung und Bewertung für alle Bilder eines Datensatzes durch.
- **Argumente:**
  - `img_dir`: Pfad zum Ordner mit Input-Bildern.
  - `gt_dir`: Pfad zum Ordner mit Ground-Truth-Masken.
  - `dataset` (optional): Name des Datensatzes (für Logging/Export).
- **Rückgabe:** `pandas.DataFrame` mit Spalten: `Bild`, `Methode`, `Dice Score`, `Datensatz` (optional).

### Ablauf:

1. **Dateinamen sammeln:** Bilddateien (`.tif`, `.png`) und GT-Dateien werden gesammelt.
2. **Matching:** Die passende GT-Datei wird anhand des Bildnamens abgeleitet.
3. **Laden & Segmentieren:** Für jedes Paar:
    - Bild und GT laden (`load_image_and_gt`).
    - Alle Methoden auf das Bild anwenden (`process_all_methods`).
    - Dice Score für jede Methode berechnen (`evaluate_segmentations`).
4. **Daten sammeln:** Ergebnisse werden gesammelt und in einem DataFrame zurückgegeben.

---

## ▶️ Anwendungsskript: Einzelordner

```python
from run_batch_evaluation import run_batch_evaluation
import os

if __name__ == "__main__":
    img_dir = "data/N2DH-GOWT1/img"
    gt_dir  = "data/N2DH-GOWT1/gt"

    df = run_batch_evaluation(img_dir, gt_dir)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/dice_scores.csv", index=False)
    print(df)
```

---

## ▶️ Anwendungsskript: Alle Datensätze

```python
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

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        os.makedirs("results", exist_ok=True)
        df_all.to_csv("results/dice_scores.csv", index=False)
        print(f"✅ Ergebnisse gespeichert: results/dice_scores.csv")
    else:
        print("⚠️ Keine Ergebnisse gesammelt.")
```

---

## 💡 Hinweise

- Es wird das Modul `tqdm` für Fortschrittsanzeigen verwendet.
- Ergebnisse werden in `results/dice_scores.csv` gespeichert.
- Dieses Skript eignet sich sowohl für Einzel- als auch Mehrfachdatensätze.



---

## 📈 Visualisierung & Analyse


# 🖼️ Visualisierung von Segmentierungen (`visualize_segmentations.py`)

Dieses Modul dient der Darstellung und Abspeicherung von Segmentierungsergebnissen aus dem Projekt zur Zellkern-Segmentierung. Es ermöglicht die vergleichende Visualisierung zwischen dem Originalbild, der Ground Truth und den Resultaten verschiedener Segmentierungsmethoden.

---

## 📌 Funktion: `visualize_segmentations(...)`

```python
def visualize_segmentations(
    image: np.ndarray,
    gt_mask: np.ndarray,
    predictions: Dict[str, np.ndarray],
    max_cols: int = 3,
    save_path: str = None
)
```

### Parameter:
- `image`: Das Originalbild in Graustufen (float, skaliert auf [0, 1])
- `gt_mask`: Die Ground-Truth-Maske als binäres Bild (boolesches Array)
- `predictions`: Ein Dictionary mit Methodenname → Segmentierungsmaske
- `max_cols`: Maximale Anzahl an Spalten in der Darstellungsübersicht (default: 3)
- `save_path`: Optionaler Dateipfad zur Abspeicherung der Visualisierung

### Ablauf:
1. Es wird eine kombinierte Liste mit Titeln und zugehörigen Bildern erstellt:
   - `"Original"`, `"Ground Truth"` und alle Einträge aus `predictions`.
2. Die Zahl der benötigten Zeilen und Spalten wird berechnet.
3. Für jede Bild-Maske-Kombination wird ein `subplot` erzeugt:
   - Das Bild wird angezeigt, der Titel gesetzt, und die Achsen werden entfernt.
4. Nicht benutzte Subplots werden deaktiviert.
5. Optional wird das Gesamtergebnis als `.png` gespeichert.

---

## 🧪 Beispielverwendung (Einzelbild)

```python
from src.load_image_pair import load_image_and_gt
from process_image import process_all_methods
from visualize_segmentation import visualize_segmentations

image, gt_mask = load_image_and_gt(
    "data/N2DH-GOWT1/img/t01.tif",
    "data/N2DH-GOWT1/gt/man_seg01.tif"
)

results = process_all_methods(image)

visualize_segmentations(image, gt_mask, results)
```

---

## 🗃️ Batch-Visualisierung aller Bilder

Ein erweiterter Codeabschnitt visualisiert und speichert automatisch die Segmentierungen aller Bilder eines Datensatzes.

### Wichtige Punkte:
- Es werden alle `.tif` und `.png` Bilder aus `img/` geladen.
- Die zugehörigen Ground-Truth-Dateien werden automatisch anhand des Dateinamens abgeleitet.
- Für jedes Bild und jede Methode werden Original, GT und Segmentierung nebeneinander angezeigt.
- Die Ergebnisse werden in `output_visuals/` gespeichert.

### Ausschnitt:

```python
for method_name, mask in predictions.items():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Originalbild")
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"Segmentierung: {method_name}")
    ...
    plt.savefig(out_path)
```

---

## ✅ Ergebnis

Die gespeicherten Bilder befinden sich in `output_visuals/{Datensatz}/{Methode}/`. Jede PNG-Datei enthält eine vergleichende Darstellung zwischen Originalbild, Ground Truth und segmentiertem Bild.

---

## 🔁 Fehlerbehandlung

Falls Ground-Truth-Dateien fehlen oder ein Bild fehlerhaft ist, wird dies mit einer Warnmeldung im Terminal angezeigt, die Verarbeitung fährt aber fort.

# 📊 Vergleich von Otsu Local Methoden (`plot_otsu_local_comparison.py`)

Dieses Skript dient der visuellen Analyse zweier Otsu-Lokal-Segmentierungsmethoden: einer eigenen Implementierung (`custom`) und einer Variante aus der `skimage`-Bibliothek. Es erzeugt einen Scatterplot, in dem für jedes Bild die Dice Scores der beiden Methoden gegenübergestellt werden.

---

## 🔢 Schrittweise Erklärung

### 1. 📥 CSV-Datei laden

```python
csv_path = os.path.join("results", "dice_scores.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")
df = pd.read_csv(csv_path)
```

- Die Datei `dice_scores.csv` enthält segmentierungsbezogene Bewertungen (Dice Scores) aller Methoden für verschiedene Bilder.
- Sie wird aus dem Ordner `results/` geladen.
- Falls sie nicht existiert, wird ein Fehler ausgelöst.

---

### 2. 🔍 Methoden filtern

```python
df_local = df[df["Methode"].isin(["Otsu Local (custom)", "Otsu Local (skimage)"])]
```

- Es werden nur die beiden Methoden `"Otsu Local (custom)"` und `"Otsu Local (skimage)"` selektiert.

---

### 3. 🔄 Tabelle umstrukturieren (Pivot)

```python
df_pivot = df_local.pivot(index="Bild", columns="Methode", values="Dice Score")
```

- `pivot` transformiert die Tabelle so, dass jede Zeile einem Bild entspricht und die beiden Spalten die Dice Scores der beiden Methoden enthalten.

---

### 4. 📈 Scatterplot erstellen

```python
plt.figure(figsize=(6, 6))
plt.scatter(df_pivot["Otsu Local (custom)"], df_pivot["Otsu Local (skimage)"], color="green", s=60)
plt.plot([0, 1], [0, 1], 'r--', label="Ideal: x = y")
plt.xlabel("Dice Score – Otsu Local (eigener Code)")
plt.ylabel("Dice Score – Otsu Local (skimage)")
plt.title("Vergleich der Otsu Local Methoden")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/otsu_local_scatterplot.png", dpi=150)
plt.show()
```

- Es wird ein 6x6 Scatterplot erzeugt.
- Jeder Punkt stellt einen Bildvergleich dar: x = eigener Code, y = skimage-Methode.
- Die rote gestrichelte Linie `x = y` zeigt die Ideallinie — Punkte auf dieser Linie deuten auf gleich gute Performance beider Methoden hin.
- Der Plot wird zusätzlich in `results/otsu_local_scatterplot.png` gespeichert.

---

## 🧪 Ziel

Der Plot zeigt die relative Performance der beiden Implementierungen bei der Segmentierung. Eine starke Korrelation entlang der Diagonalen spricht für vergleichbare Resultate. Abweichungen zeigen Unterschiede in Sensitivität oder Robustheit.

---

## 📁 Erwartete Dateistruktur

```
.
├── results/
│   └── dice_scores.csv
│   └── otsu_local_scatterplot.png  ← wird erzeugt
├── plot_otsu_local_comparison.py
```


---


# 📊 `plot_otsu_global_comparison.py`

Dieses Python-Skript erzeugt einen Scatterplot, um die Ergebnisse zweier Varianten des Otsu-Schwellenwertverfahrens zu vergleichen: die eigene Implementierung (`Otsu Global (custom)`) und die Referenzimplementierung aus `skimage` (`Otsu Global (skimage)`).

---

## 🔍 Ziel

Der Scatterplot stellt für jedes Bild den Dice Score beider Methoden gegenüber. Ein Punkt auf der Diagonalen (x=y) bedeutet, dass beide Methoden identische Ergebnisse liefern.

---

## 🧩 Code-Erklärung

### 🔁 Vorbereitung

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
```

Importiert Standardbibliotheken für Dateisystemzugriff, Datenverarbeitung und Plotten.

---

### 📂 Daten laden

```python
csv_path = os.path.join("results", "dice_scores.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")
df = pd.read_csv(csv_path)
```

Der Dice Score für jede Methode und jedes Bild wurde zuvor berechnet und in einer CSV-Datei gespeichert. Diese Datei wird hier geladen.

---

### 📊 Vergleich der Methoden

```python
df_global = df[df["Methode"].isin(["Otsu Global (custom)", "Otsu Global (skimage)"])]
df_pivot = df_global.pivot(index="Bild", columns="Methode", values="Dice Score")
```

Hier werden nur die relevanten Methoden ausgewählt und in eine vergleichbare Form gebracht (eine Zeile pro Bild, zwei Spalten für die beiden Methoden).

---

### 📈 Scatterplot erstellen

```python
plt.figure(figsize=(6, 6))
plt.scatter(
    df_pivot["Otsu Global (custom)"],
    df_pivot["Otsu Global (skimage)"],
    color="blue", s=60
)
```

Jeder Punkt im Plot steht für ein Bild. Die X- und Y-Werte sind die Dice Scores der beiden Methoden.

---

### ➕ Referenzlinie

```python
plt.plot([0, 1], [0, 1], 'r--', label="Ideal: x = y")
```

Diese diagonale Linie zeigt die ideale Übereinstimmung an. Je näher ein Punkt an dieser Linie liegt, desto ähnlicher sind die Ergebnisse beider Methoden.

---

### 🖼️ Achsen & Speicherung

```python
plt.xlabel("Dice Score – Otsu Global (eigener Code)")
plt.ylabel("Dice Score – Otsu Global (skimage)")
plt.title("Vergleich der Otsu Global Methoden")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/otsu_global_scatterplot.png", dpi=150)
plt.show()
```

Die Ausgabe wird als Bild gespeichert und angezeigt.

---

## 💡 Fazit

Mit diesem Plot kann man auf einen Blick erkennen, wie gut die eigene Otsu-Implementierung im Vergleich zur Referenzmethode abschneidet.



# 📊 Auswertung der Segmentierungsergebnisse: `plot_all_methods.py`

Dieses Skript dient der grafischen Auswertung der zuvor berechneten Dice Scores für verschiedene Segmentierungsmethoden. Es erzeugt zwei zentrale Visualisierungen:

1. **Boxplot**: Verteilung der Dice Scores pro Methode.
2. **Heatmap**: Dice Scores je Bild und Methode im Vergleich.

---

## 🧩 Aufbau des Skripts

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

- `os`: zur Dateipfadkontrolle
- `pandas`: zum Einlesen und Verarbeiten der CSV-Daten
- `matplotlib.pyplot` und `seaborn`: für die Visualisierung

---

## 📥 Dateneinlesung

```python
csv_path = os.path.join("results", "dice_scores.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")
```

- Der Pfad zur CSV-Datei mit den Dice Scores wird definiert.
- Falls die Datei nicht existiert, wird eine Fehlermeldung ausgegeben.

```python
df = pd.read_csv(csv_path)
```

- Die Daten werden in ein DataFrame geladen.

---

## 📦 Boxplot der Dice Scores pro Methode

```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Methode", y="Dice Score")
plt.xticks(rotation=45)
plt.title("Verteilung der Dice Scores je Methode")
plt.tight_layout()
plt.savefig("results/dice_scores_boxplot.png", dpi=150)
plt.show()
```

- Visualisiert die Streuung und Verteilung der Dice Scores pro Methode.
- Der Boxplot zeigt Median, Quartile und Ausreißer.
- Die Visualisierung wird als PNG gespeichert.

---

## 🔥 Heatmap der Dice Scores (Bild × Methode)

```python
df_pivot = df.pivot(index="Bild", columns="Methode", values="Dice Score")
```

- Das DataFrame wird so umgeformt, dass jede Zeile ein Bild darstellt und jede Spalte eine Methode.

```python
plt.figure(figsize=(12, 8))
sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5)
plt.title("Heatmap der Dice Scores (Bilder × Methoden)")
plt.tight_layout()
plt.savefig("results/dice_scores_heatmap.png", dpi=150)
plt.show()
```

- Eine annotierte Heatmap zeigt, wie gut jede Methode auf jedem Bild abgeschnitten hat.
- Farbgebung über das `viridis`-Farbschema.
- Auch diese Abbildung wird als PNG gespeichert.

---

## 📁 Ergebnis

Nach erfolgreichem Durchlauf befinden sich im Verzeichnis `results/`:

- `dice_scores_boxplot.png`: Vergleichende Übersicht über die Verteilung der Methoden.
- `dice_scores_heatmap.png`: Detaillierter Vergleich pro Bild.

Diese Darstellungen erleichtern es, systematisch Schwächen und Stärken der einzelnen Segmentierungsmethoden zu erkennen.

---

## 📌 Hinweise

- Stelle sicher, dass `results/dice_scores.csv` bereits durch vorherige Skripte erzeugt wurde.
- Die Datei muss mindestens die Spalten `Bild`, `Methode` und `Dice Score` enthalten.




### `plot_dice_score_summary.py`
- Nutzt `dice_scores.csv`
- Erstellt:
  - Boxplot für jede Methode
  - Heatmap für Dice-Werte je Bild & Methode

### `plot_otsu_scatter.py`
- Scatterplot-Vergleich zwischen:
  - Otsu Global (custom)
  - Otsu Global (skimage)
- Zeigt Korrelation der Ergebnisse

---

## 📓 Notebook

### `Otsu_Segmentierung_Auswertung.ipynb`
- Interaktive Darstellung der Methoden und Ergebnisse
- Kombiniert Code + Erläuterung + Plot-Ausgaben

---

## 📎 Hinweise zur Nutzung

- Stelle sicher, dass alle Ordnerstrukturen wie oben beschrieben vorhanden sind
- Führe `run_all_evaluations.py` aus, um alle Methoden zu bewerten
- Die Resultate sind anschließend in `results/` und `output_visuals/` zu finden

---

## 📬 Fragen & Weiterführung

Dieses Projekt bietet eine solide Grundlage zur Analyse und zum Vergleich von Segmentierungsverfahren. Für weiterführende Projekte können neuronale Netzwerke, z. B. U-Net, hinzugezogen werden. Feedback und Erweiterungen sind willkommen!
