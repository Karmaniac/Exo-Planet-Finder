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

## Requirements

```bash
python -m venv exoplanet
exoplanet\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## Usage (Skip to #5 to use the pretrained model with specs given above)

### Step 1 — Fetch labeled dataset
```bash
python fetch_tess_labeled_dataset.py
python fetch_astronet_dataset.py --exclude-existing labeled_tess_dataset.csv
```

### Step 2 — Download & cache light curves
```bash
python train_classifier.py download --csv labeled_tess_dataset.csv
python train_classifier.py download --csv astronet_tess_dataset.csv --cache lc_cache
```

### Step 3 — Merge the datasets
```bash
python -c "import pandas as pd; a=pd.read_csv('labeled_tess_dataset.csv'); b=pd.read_csv('astronet_tess_dataset.csv'); merged=pd.concat([a,b]).drop_duplicates(subset='TIC_ID').reset_index(drop=True); merged.to_csv('labeled_tess_dataset.csv',index=False)"
```

### Step 4 — Train the model
```bash
python train_classifier.py train --epochs 200 --patience 30 --tess-csv labeled_tess_dataset.csv
```

### Step 5 — Classify a new target
```bash
python inference.py <TIC_ID> --csv labeled_tess_dataset.csv
```

---

## Labels

| Label | Meaning |
|---|---|
| `1` | Confirmed planet |
| `0` | False positive |
