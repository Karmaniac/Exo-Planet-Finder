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

## Usage

### Step 1 — Fetch labeled dataset
```bash
python fetch_tess_labeled_dataset.py
```

### Step 2 — Download & cache light curves
```bash
python train_classifier.py download --csv labeled_tess_dataset.csv
```

### Step 3 — Train the model
```bash
python train_classifier.py train --epochs 200 --patience 30 --tess-csv labeled_tess_dataset.csv
```

### Step 4 — Classify a new target
```bash
python inference.py <TIC_ID> --csv labeled_tess_dataset.csv
```

### Verify saved model
```bash
python eval_model.py --csv labeled_tess_dataset.csv
```

---

## Labels

| Label | Meaning |
|---|---|
| `1` | Confirmed planet |
| `0` | False positive |

---

## Notes

- Light curves are downloaded from MAST via SPOC pipeline (QLP fallback). Up to 3 sectors per target.
- The download step is resumable — interrupted runs skip already-cached targets.
- Cache directories (`lc_cache/`, `kepler_cache/`) and model weights (`exoplanet_cnn.pt`) are gitignored.
