"""
train_classifier.py
───────────────────
Trains a Random Forest + Gradient Boosting classifier ensemble to distinguish
genuine exoplanet transits from false positives using REAL BLS features
extracted from TESS light curves.

Improvements over v1
────────────────────
  1. More targets supported — fetch with --max-targets 1000+
  2. Derived feature: depth_over_duration (sharper signal for real transits)
  3. Both RF and GradientBoosting trained and compared — best one saved
  4. BLS averaged across ALL available sectors per target (not just sector[0])

Feature extraction
──────────────────
  bls_period          – mean best BLS period across sectors (days)
  bls_duration        – mean best transit duration across sectors (days)
  bls_depth           – mean fractional flux drop across sectors
  bls_snr             – mean signal-to-noise across sectors
  bls_period_std      – std of period across sectors (stable = low std)
  depth_over_duration – bls_depth / bls_duration (sharpness proxy)
  toi_period          – period from TOI table (days)
  toi_depth           – depth from TOI table (fractional)
  toi_duration        – duration from TOI table (days)
  n_sectors           – number of sectors available

Labels
──────
  1 = confirmed / known planet
  0 = false positive

Outputs
───────
  exoplanet_rf_classifier.joblib  – best model (RF or GBT, whichever scores higher)

Requirements
────────────
  pip install lightkurve pandas numpy scikit-learn joblib requests astroquery

Usage
─────
  python train_classifier.py --csv labeled_tess_dataset.csv
  python train_classifier.py --csv labeled_tess_dataset.csv --max-targets 200
  python train_classifier.py --csv labeled_tess_dataset.csv --toi-only
"""

import argparse
import ast
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
MODEL_PATH  = "exoplanet_rf_classifier.joblib"

# Coarse BLS grid for training — fast but good enough for feature extraction.
# test.py uses a finer grid for accurate single-target inference.
BLS_PERIODS   = np.linspace(0.5, 25, 500)
BLS_DURATIONS = np.linspace(0.01, 0.3, 10)

# Max sectors to average BLS over per target (cap to avoid very long runs)
MAX_SECTORS_PER_TARGET = 3


# ── Feature extraction ────────────────────────────────────────────────────────

def run_bls_single_sector(tic_id: str, sector: int) -> dict | None:
    """
    Download the SPOC light curve for one sector and run BLS.
    Returns a dict of BLS features, or None on failure.
    """
    try:
        import lightkurve as lk

        results = lk.search_lightcurve(
            f"TIC {tic_id}", mission="TESS", author="SPOC", sector=sector
        )
        if results is None or len(results) == 0:
            return None

        lc = results[0].download(quality_bitmask="hard")
        if lc is None:
            return None

        lc = lc.remove_nans().remove_outliers(sigma=5).normalize()

        bls = lc.to_periodogram(method="bls", period=BLS_PERIODS, duration=BLS_DURATIONS)

        best_period   = float(bls.period_at_max_power.value)
        best_duration = float(bls.duration_at_max_power.value)
        best_depth    = float(bls.depth_at_max_power.value)
        snr           = float(bls.snr.max())

        return {
            "bls_period":   best_period,
            "bls_duration": best_duration,
            "bls_depth":    best_depth,
            "bls_snr":      snr,
        }
    except Exception:
        return None


def run_bls_all_sectors(tic_id: str, sectors: list) -> dict | None:
    """
    Run BLS across all available sectors (up to MAX_SECTORS_PER_TARGET)
    and return averaged features. A stable period across sectors is a
    strong indicator of a real transit signal.
    """
    sectors_to_use = sectors[:MAX_SECTORS_PER_TARGET]
    results = []

    for sector in sectors_to_use:
        r = run_bls_single_sector(tic_id, sector)
        if r is not None:
            results.append(r)
        time.sleep(0.05)

    if not results:
        return None

    avg = {
        "bls_period":     float(np.mean([r["bls_period"]   for r in results])),
        "bls_duration":   float(np.mean([r["bls_duration"] for r in results])),
        "bls_depth":      float(np.mean([r["bls_depth"]    for r in results])),
        "bls_snr":        float(np.mean([r["bls_snr"]      for r in results])),
        # Std of period across sectors — real planets are consistent, FPs vary
        "bls_period_std": float(np.std([r["bls_period"]    for r in results])) if len(results) > 1 else 0.0,
    }
    return avg


def extract_features_for_row(row: pd.Series, use_bls: bool) -> dict:
    """
    Build a feature dict for one TIC target.
    Averages BLS over all available sectors. Falls back to TOI values on failure.
    """
    tic_id    = str(row["TIC_ID"])
    sectors   = ast.literal_eval(str(row["sectors"])) if "sectors" in row and pd.notna(row.get("sectors")) else []
    n_sectors = int(row.get("n_sectors", len(sectors)))

    toi_period   = float(row["period"])            if "period"      in row and pd.notna(row.get("period"))      else np.nan
    toi_depth    = float(row["depth_ppm"]) / 1e6   if "depth_ppm"   in row and pd.notna(row.get("depth_ppm"))   else np.nan
    toi_duration = float(row["duration_hr"]) / 24  if "duration_hr" in row and pd.notna(row.get("duration_hr")) else np.nan

    feat = {
        "bls_period":          np.nan,
        "bls_duration":        np.nan,
        "bls_depth":           np.nan,
        "bls_snr":             np.nan,
        "bls_period_std":      np.nan,
        "depth_over_duration": np.nan,
        "toi_period":          toi_period,
        "toi_depth":           toi_depth,
        "toi_duration":        toi_duration,
        "n_sectors":           n_sectors,
    }

    if use_bls and sectors:
        bls = run_bls_all_sectors(tic_id, sectors)
        if bls:
            feat.update({
                "bls_period":          bls["bls_period"],
                "bls_duration":        bls["bls_duration"],
                "bls_depth":           bls["bls_depth"],
                "bls_snr":             bls["bls_snr"],
                "bls_period_std":      bls["bls_period_std"],
                "depth_over_duration": bls["bls_depth"] / (bls["bls_duration"] + 1e-6),
            })

    return feat


def build_feature_matrix(df: pd.DataFrame, use_bls: bool):
    feature_rows = []
    labels       = []
    total        = len(df)
    bls_ok_count = 0

    print(f"\nExtracting features for {total:,} targets (BLS={'ON' if use_bls else 'OFF'})...")
    if use_bls:
        print(f"  Averaging BLS over up to {MAX_SECTORS_PER_TARGET} sectors per target.\n")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 25 == 0 or i == 1:
            print(f"  {i}/{total}  (BLS successes: {bls_ok_count})")

        feat = extract_features_for_row(row, use_bls)

        if use_bls and not np.isnan(feat["bls_period"]):
            bls_ok_count += 1

        feature_rows.append(feat)
        labels.append(int(row["label"]))

        if use_bls:
            time.sleep(0.05)

    if use_bls:
        print(f"\n  BLS succeeded on {bls_ok_count}/{total} targets "
              f"({100*bls_ok_count/total:.1f}%). TOI fallback used for the rest.")

    feature_names = [
        "bls_period", "bls_duration", "bls_depth", "bls_snr", "bls_period_std",
        "depth_over_duration",
        "toi_period", "toi_depth", "toi_duration", "n_sectors",
    ]

    X = np.array([[row[f] for f in feature_names] for row in feature_rows])
    y = np.array(labels)
    return X, y, feature_names


# ── Build and compare models ──────────────────────────────────────────────────

def build_models():
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])

    gbt = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_SEED,
        )),
    ])

    return {"RandomForest": rf, "GradientBoosting": gbt}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train exoplanet classifier on real TESS data.")
    parser.add_argument("--csv",          type=str,  default="labeled_tess_dataset.csv")
    parser.add_argument("--max-targets",  type=int,  default=None)
    parser.add_argument("--toi-only",     action="store_true")
    parser.add_argument("--model-output", type=str,  default=MODEL_PATH)
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: '{csv_path}' not found. Run fetch_tess_labeled_dataset.py first.")
        return

    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts()
    print(f"Loaded {len(df):,} rows  |  Planets: {counts.get(1,0):,}  FPs: {counts.get(0,0):,}")

    if args.max_targets:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.max_targets // 2), random_state=RANDOM_SEED)
        ).reset_index(drop=True)
        print(f"Subsampled to {len(df):,} targets (stratified).")

    # ── Extract features ──────────────────────────────────────────────────────
    X, y, feature_names = build_feature_matrix(df, use_bls=not args.toi_only)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Cross-validate both models ────────────────────────────────────────────
    models     = build_models()
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    best_name  = None
    best_score = -1
    best_model = None

    print("\n── Cross-validation (5-fold F1) ─────────────────────")
    for name, pipeline in models.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        mean_f1 = scores.mean()
        print(f"  {name:<20} {scores.round(3)}  mean={mean_f1:.3f} ± {scores.std():.3f}")
        if mean_f1 > best_score:
            best_score = mean_f1
            best_name  = name
            best_model = pipeline

    print(f"\n  Best model: {best_name} (F1={best_score:.3f})")

    # ── Final evaluation on held-out test set ─────────────────────────────────
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print("\n── Classification Report ────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["False Positive", "Planet Candidate"]))

    print("── Confusion Matrix ─────────────────────────────────")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted FP  Predicted Planet")
    print(f"  Actual FP          {cm[0,0]:>5}          {cm[0,1]:>5}")
    print(f"  Actual Planet      {cm[1,0]:>5}          {cm[1,1]:>5}")

    print("\n── Feature Importances ──────────────────────────────")
    importances = best_model.named_steps["clf"].feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {name:<22} {imp:.4f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(best_model, args.model_output)
    print(f"\nSaved {best_name} model to '{args.model_output}'")
    print(f"\nFeature order for inference:")
    print(f"  {feature_names}")


if __name__ == "__main__":
    main()
