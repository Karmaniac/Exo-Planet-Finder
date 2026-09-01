# ExoPlanetFinder

A 1D CNN that classifies TESS light curves as planet candidates or false positives, trained on labeled targets from the NASA TOI catalog and Astronet-Triage dataset.

**Results:** 86% false positive precision · 79% accuracy · 4,301 training targets

---

## Architecture

```
Light curve branch : 264-point vector (global view · local view · sec eclipse · even/odd diff)
                     4× Conv1d blocks (BatchNorm · ReLU · MaxPool) → Flatten → 1024

Scalar branch      : log(period) · log(depth) · duration → 16

Merge              : 1040 → Linear 256 → Linear 64 → Sigmoid → planet probability
```

---

## Layout

```
gui.py, inference.py, train_classifier.py   # entry points
fetch_tess_labeled_dataset.py, fetch_astronet_dataset.py

data/     labeled_tess_dataset.csv, astronet_tess_dataset.csv
models/   exoplanet_cnn.pt, exoplanet_cnn_meta.json
paper/    write-up and slides
```

`lc_cache/` (downloaded light curves) and `stellar_cache.json` (TIC catalog
lookups) are created on first run and gitignored.

---

## Requirements

```bash
python -m venv exoplanet
source exoplanet/bin/activate      # Linux/Mac
exoplanet\Scripts\activate         # Windows
pip install -r requirements.txt
```

---

## Usage (Skip to #5 to use the pretrained model with specs given above)

### Step 1 — Fetch labeled dataset
```bash
python fetch_tess_labeled_dataset.py
python fetch_astronet_dataset.py --exclude-existing data/labeled_tess_dataset.csv
```

### Step 2 — Download & cache light curves
```bash
python train_classifier.py download --csv data/labeled_tess_dataset.csv
python train_classifier.py download --csv data/astronet_tess_dataset.csv --cache lc_cache
```

### Step 3 — Merge the datasets
```bash
python -c "import pandas as pd; a=pd.read_csv('data/labeled_tess_dataset.csv'); b=pd.read_csv('data/astronet_tess_dataset.csv'); merged=pd.concat([a,b]).drop_duplicates(subset='TIC_ID').reset_index(drop=True); merged.to_csv('data/labeled_tess_dataset.csv',index=False)"
```

### Step 4 — Train the model
```bash
python train_classifier.py train --epochs 200 --patience 30 --tess-csv data/labeled_tess_dataset.csv
```

### Step 5 — Classify a new target
```bash
python inference.py <TIC_ID> --csv data/labeled_tess_dataset.csv
```

### Step 6 — Or classify with the GUI
```bash
pip install torch numpy pandas astroquery matplotlib   # + python3-tk (system package, Linux)
python gui.py
```
Enter a TIC ID and hit Classify. Shows the CNN's planet/false-positive call alongside
estimated physical properties (radius, mass, density, semi-major axis, equilibrium
temperature, insolation) pulled from the MAST TIC catalog, plus the folded light curve.

---

## Labels

| Label | Meaning |
|---|---|
| `1` | Confirmed planet |
| `0` | False positive |
