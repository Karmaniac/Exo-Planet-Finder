"""
train_classifier.py
───────────────────
1D CNN exoplanet transit classifier (Shallue & Vanderburg 2018 inspired).

Two-phase workflow
──────────────────
  Phase 1 — Download & cache processed light curves (network-heavy, slow):

    python train_classifier.py download --csv labeled_tess_dataset.csv
    python train_classifier.py download --csv labeled_tess_dataset.csv --max-targets 200

    Outputs a cache directory (default: lc_cache/) containing:
      <TIC_ID>.npy   — 262-element float32 vector per target
      cache_labels.csv — TIC_ID, label, mission columns

  Phase 2 — Train from cache (fast, no network needed):

    python train_classifier.py train
    python train_classifier.py train --cache lc_cache --epochs 80 --batch-size 64
    python train_classifier.py train --pretrain-cache kepler_cache --cache lc_cache

    You can retrain as many times as you want without re-downloading.

Architecture
────────────
  Input : 201-bin global view + 61-bin local view  →  262-point vector
  Layers: Conv1d blocks → MaxPool → Flatten → FC → Dropout → Sigmoid
  Output: planet probability

Requirements
────────────
  pip install lightkurve pandas numpy torch scikit-learn requests astroquery
"""

import argparse
import ast
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

RANDOM_SEED   = 42
MODEL_PATH    = "exoplanet_cnn.pt"
META_PATH     = "exoplanet_cnn_meta.json"
GLOBAL_BINS   = 201
LOCAL_BINS    = 61
INPUT_SIZE    = GLOBAL_BINS + LOCAL_BINS + 2   # 264 (+ secondary eclipse depth, even/odd diff)
BLS_PERIODS   = np.linspace(0.5, 25, 500)
BLS_DURATIONS = np.linspace(0.01, 0.3, 10)
MAX_SECTORS   = 3
DEFAULT_CACHE = "lc_cache"

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Light curve processing ─────────────────────────────────────────────────────

def process_lightcurve(tic_id: str, sectors: list, mission: str = "TESS") -> np.ndarray | None:
    """
    Download up to MAX_SECTORS light curves, concatenate, run BLS, fold and bin
    into a (262,) numpy array. Returns None on failure.
    """
    try:
        import lightkurve as lk
        from astropy.timeseries import BoxLeastSquares

        all_times, all_fluxes = [], []

        for sector in sectors[:MAX_SECTORS]:
            try:
                if mission == "Kepler":
                    if sector == 1:
                        res = lk.search_lightcurve(f"KIC {tic_id}", mission="Kepler")
                    else:
                        continue
                else:
                    # Try SPOC first, fall back to QLP (used by Astronet/MIT)
                    res = lk.search_lightcurve(
                        f"TIC {tic_id}", mission="TESS", author="SPOC", sector=sector
                    )
                    if res is None or len(res) == 0:
                        res = lk.search_lightcurve(
                            f"TIC {tic_id}", mission="TESS", author="QLP", sector=sector
                        )
                if res is None or len(res) == 0:
                    continue
                lc = res[0].download(quality_bitmask="default")
                if lc is None:
                    continue
                lc = lc.remove_nans().remove_outliers(sigma=5).flatten(window_length=401).normalize()
                all_times.append(lc.time.value)
                all_fluxes.append(lc.flux.value)
                time.sleep(0.05)
            except Exception:
                continue

        if not all_times:
            return None

        times  = np.concatenate(all_times)
        fluxes = np.concatenate(all_fluxes)

        bls   = BoxLeastSquares(times, fluxes)
        power = bls.power(BLS_PERIODS, BLS_DURATIONS)

        best_idx      = np.argmax(power.power)
        best_period   = float(power.period[best_idx])
        best_t0       = float(power.transit_time[best_idx])
        best_duration = float(power.duration[best_idx])

        phase = ((times - best_t0) / best_period) % 1.0
        phase[phase > 0.5] -= 1.0
        sort_idx = np.argsort(phase)
        phase    = phase[sort_idx]
        fluxes   = fluxes[sort_idx]

        global_bins = np.linspace(-0.5, 0.5, GLOBAL_BINS + 1)
        global_view = _bin_lc(phase, fluxes, global_bins)

        half_width = max(best_duration * 2.0, 0.02)
        local_bins = np.linspace(-half_width, half_width, LOCAL_BINS + 1)
        local_view = _bin_lc(phase, fluxes, local_bins)

        global_view = _normalise(global_view)
        local_view  = _normalise(local_view)

        # ── Extra features ────────────────────────────────────────────────────
        # 1. Secondary eclipse depth at phase 0.5 (eclipsing binaries show a dip here)
        sec_half = max(best_duration * 2.0, 0.02)
        sec_mask = (phase >= 0.5 - sec_half) & (phase <= 0.5 + sec_half)
        if sec_mask.sum() > 0:
            sec_depth = float(1.0 - np.median(fluxes[sec_mask]))
        else:
            sec_depth = 0.0

        # 2. Even/odd transit depth difference (eclipsing binaries alternate depth)
        transit_mask = (phase >= -half_width) & (phase <= half_width)
        t_times  = times[sort_idx][transit_mask]
        t_fluxes = fluxes[transit_mask]
        if len(t_times) > 4:
            transit_nums = np.floor((t_times - best_t0) / best_period).astype(int)
            even_flux = t_fluxes[transit_nums % 2 == 0]
            odd_flux  = t_fluxes[transit_nums % 2 == 1]
            even_depth = float(1.0 - np.median(even_flux)) if len(even_flux) > 0 else 0.0
            odd_depth  = float(1.0 - np.median(odd_flux))  if len(odd_flux)  > 0 else 0.0
            even_odd_diff = abs(even_depth - odd_depth)
        else:
            even_odd_diff = 0.0

        extra = np.array([sec_depth, even_odd_diff], dtype=np.float32)

        return np.concatenate([global_view, local_view, extra]).astype(np.float32)

    except Exception:
        return None


def _bin_lc(phase, flux, bins):
    n = len(bins) - 1
    out = np.ones(n, dtype=np.float32)
    for i in range(n):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        if mask.sum() > 0:
            out[i] = float(np.median(flux[mask]))
    return out


def _normalise(v):
    med, std = np.median(v), np.std(v)
    if std < 1e-8:
        return np.zeros_like(v)
    return (v - med) / std


# ── Phase 1: Download & cache ──────────────────────────────────────────────────

def cmd_download(args):
    """
    Download and cache processed light curve vectors to disk.
    Skips targets that are already cached (safe to resume after interruption).
    """
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: '{csv_path}' not found.")
        return

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "mission" not in df.columns:
        df["mission"] = "TESS"

    counts = df["label"].value_counts()
    print(f"Loaded {len(df):,} rows  |  Planets: {counts.get(1,0):,}  FPs: {counts.get(0,0):,}")

    if args.max_targets:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.max_targets // 2), random_state=RANDOM_SEED)
        ).reset_index(drop=True)
        print(f"Subsampled to {len(df):,} targets (stratified).")

    labels_path = cache_dir / "cache_labels.csv"
    if labels_path.exists():
        existing = pd.read_csv(labels_path)
        cached_ids = set(existing["TIC_ID"].astype(str))
        print(f"Resuming — {len(cached_ids)} targets already cached, skipping them.")
    else:
        existing  = pd.DataFrame(columns=["TIC_ID", "label", "mission"])
        cached_ids = set()

    total = len(df)
    new_rows = []
    ok = skipped = failed = 0

    print(f"\nDownloading {total:,} targets → {cache_dir}/\n")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        tic_id  = str(row["TIC_ID"])
        label   = int(row["label"])
        mission = str(row.get("mission", "TESS"))

        if tic_id in cached_ids:
            skipped += 1
            continue

        if i % 25 == 0 or i == 1:
            print(f"  {i}/{total}  ok={ok}  skipped={skipped}  failed={failed}")

        sectors = []
        if pd.notna(row.get("sectors", None)):
            try:
                sectors = ast.literal_eval(str(row["sectors"]))
            except Exception:
                pass

        vec = process_lightcurve(tic_id, sectors, mission=mission)

        if vec is not None:
            np.save(cache_dir / f"{tic_id}.npy", vec)
            new_rows.append({"TIC_ID": tic_id, "label": label, "mission": mission})
            ok += 1
        else:
            failed += 1

    # Append new rows to labels CSV
    if new_rows:
        updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        updated.to_csv(labels_path, index=False)

    total_cached = len(existing) + len(new_rows)
    print(f"\nDone.  New: {ok}  Skipped: {skipped}  Failed: {failed}")
    print(f"Cache: {cache_dir}/  ({total_cached} total targets)")
    print(f"Labels: {labels_path}")


# ── Load from cache ────────────────────────────────────────────────────────────

def load_cache(cache_dir: Path, scalar_csv: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cached .npy vectors, labels, and optional normalized scalar features."""
    labels_path = cache_dir / "cache_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"No cache_labels.csv found in '{cache_dir}'. Run 'download' first.")

    label_df = pd.read_csv(labels_path)
    label_df["TIC_ID"] = label_df["TIC_ID"].astype(str)

    scalar_map = {}
    if scalar_csv and Path(scalar_csv).exists():
        sdf = pd.read_csv(scalar_csv)
        sdf["TIC_ID"] = sdf["TIC_ID"].astype(str)
        for _, row in sdf.iterrows():
            period   = float(row.get("period",      0.0) or 0.0)
            depth    = float(row.get("depth_ppm",   0.0) or 0.0)
            duration = float(row.get("duration_hr", 0.0) or 0.0)
            scalar_map[str(row["TIC_ID"])] = [np.log1p(period), np.log1p(depth), duration]

    X, y, S = [], [], []
    missing = 0

    for _, row in label_df.iterrows():
        tic_id   = str(row["TIC_ID"])
        npy_path = cache_dir / f"{tic_id}.npy"
        if npy_path.exists():
            X.append(np.load(npy_path))
            y.append(int(row["label"]))
            S.append(scalar_map.get(tic_id, [0.0, 0.0, 0.0]))
        else:
            missing += 1

    if missing:
        print(f"  Warning: {missing} entries in labels CSV have no matching .npy file (skipped).")

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int64)
    S_arr = np.array(S, dtype=np.float32)

    # Replace NaN/inf with 0 before normalizing
    S_arr = np.nan_to_num(S_arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize scalars to zero mean, unit std
    if S_arr.shape[0] > 0 and S_arr.std() > 1e-8:
        s_mean = S_arr.mean(axis=0)
        s_std  = S_arr.std(axis=0) + 1e-8
        S_arr  = (S_arr - s_mean) / s_std

    return X_arr, y_arr, S_arr


# ── CNN model ──────────────────────────────────────────────────────────────────

N_SCALARS = 3  # log(period), log(depth_ppm), duration_hr

class ExoplanetCNN(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE, dropout: float = 0.55, n_scalars: int = N_SCALARS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),  nn.BatchNorm1d(16),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.scalar_branch = nn.Sequential(
            nn.Linear(n_scalars, 16), nn.ReLU(),
            nn.Linear(16, 16),        nn.ReLU(),
        )
        self.fc = nn.Sequential(
        nn.Linear(128 * 8 + 16, 256), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(256, 64),            nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(64, 1),  # no Sigmoid — BCEWithLogitsLoss handles it
)

    def forward(self, x, s):
        conv_out   = self.conv(x).flatten(1)
        scalar_out = self.scalar_branch(s)
        merged     = torch.cat([conv_out, scalar_out], dim=1)
        out = self.fc(merged).squeeze(1)
        return torch.sigmoid(out).clamp(1e-7, 1 - 1e-7)


# ── Training loop ──────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, S_train, S_val, epochs, batch_size, lr, init_model=None, patience=15):
    model = ExoplanetCNN().to(DEVICE)
    if init_model is not None:
        model.load_state_dict(init_model.state_dict())
        print("  Loaded pretrained weights.")

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    print(f"  Class weights: pos_weight={pos_weight.item():.3f}  "
          f"(neg={n_neg} pos={n_pos})")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train).unsqueeze(1),
            torch.tensor(S_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=True
    )
    X_v = torch.tensor(X_val).unsqueeze(1).to(DEVICE)
    S_v = torch.tensor(S_val, dtype=torch.float32).to(DEVICE)
    y_v = torch.tensor(y_val, dtype=torch.float32)

    best_f1, best_weights = 0.0, None
    epochs_no_improve = 0

    # Sanity check inputs
    assert not np.isnan(X_train).any(), "NaN in X_train"
    assert not np.isnan(S_train).any(), "NaN in S_train"
    print(f"Training  |  {len(X_train)} train  {len(X_val)} val  "
          f"|  epochs={epochs}  batch={batch_size}  lr={lr}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, sb, yb in train_loader:
            xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb, sb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            preds = (model(X_v, S_v).cpu().numpy() >= 0.5).astype(int)

        tp = int(((preds == 1) & (y_v.numpy() == 1)).sum())
        fp = int(((preds == 1) & (y_v.numpy() == 0)).sum())
        fn = int(((preds == 0) & (y_v.numpy() == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)

        scheduler.step(f1)

        if f1 > best_f1:
            best_f1          = f1
            best_weights     = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"loss={total_loss/len(X_train):.4f}  "
                  f"prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}"
                  + ("  *" if f1 == best_f1 else ""))

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\n  Best val F1: {best_f1:.3f}")
    model.load_state_dict(best_weights)
    return model


# ── Phase 2: Train ─────────────────────────────────────────────────────────────

def cmd_train(args):
    """Train (or retrain) the CNN entirely from cached .npy files — no downloads."""
    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Optional: pretrain on Kepler cache first
    pretrained_model = None
    if args.pretrain_cache:
        pretrain_dir = Path(args.pretrain_cache)
        print(f"\nLoading Kepler pretrain cache from '{pretrain_dir}'...")
        X_k, y_k, S_k = load_cache(pretrain_dir, scalar_csv=args.kepler_csv)
        k_counts  = np.bincount(y_k)
        print(f"Kepler: {len(X_k):,} samples  |  Planets: {k_counts[1]:,}  FPs: {k_counts[0]:,}")

        if len(X_k) >= 20:
            # Balance Kepler to 1:1 planet/FP ratio to avoid bias toward FPs
            planet_idx = np.where(y_k == 1)[0]
            fp_idx     = np.where(y_k == 0)[0]
            n_each     = min(len(planet_idx), len(fp_idx))
            rng        = np.random.default_rng(RANDOM_SEED)
            balanced_idx = np.concatenate([
                rng.choice(planet_idx, n_each, replace=False),
                rng.choice(fp_idx,     n_each, replace=False),
            ])
            rng.shuffle(balanced_idx)
            X_k, y_k, S_k = X_k[balanced_idx], y_k[balanced_idx], S_k[balanced_idx]
            print(f"Kepler balanced: {n_each} planets + {n_each} FPs = {len(X_k):,} total")

            X_ktr, X_ktmp, y_ktr, y_ktmp, S_ktr, S_ktmp = train_test_split(
                X_k, y_k, S_k, test_size=0.15, random_state=RANDOM_SEED, stratify=y_k
            )
            X_ktr, X_kval, y_ktr, y_kval, S_ktr, S_kval = train_test_split(
                X_ktr, y_ktr, S_ktr, test_size=0.15, random_state=RANDOM_SEED, stratify=y_ktr
            )
            print(f"\n-- Phase 1: Pretraining on Kepler ({len(X_ktr):,} samples) --")
            pretrained_model = train_model(
                X_ktr, y_ktr, X_kval, y_kval, S_ktr, S_kval,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                patience=args.patience
            )
        else:
            print("Not enough Kepler samples — skipping pretrain.")

    # Load TESS cache
    cache_dir = Path(args.cache)
    print(f"\nLoading TESS cache from '{cache_dir}'...")
    X, y, S = load_cache(cache_dir, scalar_csv=args.tess_csv)
    counts = np.bincount(y)
    print(f"Loaded {len(X):,} samples  |  Planets: {counts[1]:,}  FPs: {counts[0]:,}")

    if len(X) < 20:
        print("ERROR: Not enough cached samples to train.")
        return

    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
        X, y, S, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val, S_train, S_val = train_test_split(
        X_train, y_train, S_train, test_size=0.15, random_state=RANDOM_SEED, stratify=y_train
    )
    print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}\n")

    if pretrained_model is not None:
        print("-- Phase 2: Fine-tuning on TESS (lr reduced 10x) --")
        model = train_model(
            X_train, y_train, X_val, y_val, S_train, S_val,
            epochs=max(args.epochs, 50),
            batch_size=args.batch_size,
            lr=args.lr / 10,
            init_model=pretrained_model,
            patience=args.patience,
        )
    else:
        model = train_model(
            X_train, y_train, X_val, y_val, S_train, S_val,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            patience=args.patience,
        )

    # Evaluate on held-out test set
    model.eval()
    with torch.no_grad():
        probs = model(
            torch.tensor(X_test).unsqueeze(1).to(DEVICE),
            torch.tensor(S_test, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    print("\n── Classification Report ────────────────────────────")
    print(classification_report(y_test, preds, target_names=["False Positive", "Planet Candidate"]))

    print("── Confusion Matrix ─────────────────────────────────")
    cm = confusion_matrix(y_test, preds)
    print(f"                 Predicted FP  Predicted Planet")
    print(f"  Actual FP          {cm[0,0]:>5}          {cm[0,1]:>5}")
    print(f"  Actual Planet      {cm[1,0]:>5}          {cm[1,1]:>5}")

    # Save model + metadata
    torch.save(model.state_dict(), args.model_output)
    meta = {
        "input_size":    INPUT_SIZE,
        "global_bins":   GLOBAL_BINS,
        "local_bins":    LOCAL_BINS,
        "bls_periods":   [float(BLS_PERIODS[0]),   float(BLS_PERIODS[-1]),   len(BLS_PERIODS)],
        "bls_durations": [float(BLS_DURATIONS[0]), float(BLS_DURATIONS[-1]), len(BLS_DURATIONS)],
        "max_sectors":   MAX_SECTORS,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel    : {args.model_output}")
    print(f"Metadata : {META_PATH}")
    print(f"Device   : {DEVICE}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exoplanet CNN — two-phase workflow: download then train.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download & cache light curves
  python train_classifier.py download --csv labeled_tess_dataset.csv
  python train_classifier.py download --csv labeled_tess_dataset.csv --max-targets 200

  # Train from cache (repeatable, no network needed)
  python train_classifier.py train
  python train_classifier.py train --epochs 80 --batch-size 64

  # Pretrain on Kepler cache, fine-tune on TESS cache
  python train_classifier.py download --csv labeled_kepler_dataset.csv --cache kepler_cache
  python train_classifier.py train --pretrain-cache kepler_cache --cache lc_cache
"""
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── download subcommand ──
    dl = sub.add_parser("download", help="Fetch & cache processed light curves to disk.")
    dl.add_argument("--csv",         type=str, default="labeled_tess_dataset.csv",
                    help="Labeled dataset CSV (default: labeled_tess_dataset.csv)")
    dl.add_argument("--cache",       type=str, default=DEFAULT_CACHE,
                    help=f"Cache directory (default: {DEFAULT_CACHE})")
    dl.add_argument("--max-targets", type=int, default=None,
                    help="Limit number of targets (stratified, for quick tests)")

    # ── train subcommand ──
    tr = sub.add_parser("train", help="Train CNN from cached light curves (no network needed).")
    tr.add_argument("--cache",          type=str,   default=DEFAULT_CACHE,
                    help=f"TESS cache directory (default: {DEFAULT_CACHE})")
    tr.add_argument("--pretrain-cache", type=str,   default=None,
                    help="Kepler cache directory for pretraining (optional)")
    tr.add_argument("--epochs",         type=int,   default=50)
    tr.add_argument("--batch-size",     type=int,   default=32)
    tr.add_argument("--lr",             type=float, default=1e-3)
    tr.add_argument("--tess-csv",        type=str,   default="labeled_tess_dataset.csv",
                    help="TESS labeled CSV for scalar features")
    tr.add_argument("--kepler-csv",      type=str,   default="labeled_kepler_dataset.csv",
                    help="Kepler labeled CSV for scalar features")
    tr.add_argument("--patience",        type=int,   default=15,
                    help="Early stopping patience in epochs (default: 15)")
    tr.add_argument("--model-output",   type=str,   default=MODEL_PATH)

    args = parser.parse_args()

    if args.command == "download":
        cmd_download(args)
    elif args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()