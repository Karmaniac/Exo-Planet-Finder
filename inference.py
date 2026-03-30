"""
inference.py
────────────
Run the trained CNN exoplanet classifier on a new TESS target.

Usage:
  python inference.py 231663901
  python inference.py 231663901 --model exoplanet_cnn.pt
  python inference.py 231663901 --period 4.41 --depth 15062 --duration 3.65
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ── Must match train_classifier.py exactly ────────────────────────────────────
GLOBAL_BINS   = 201
LOCAL_BINS    = 61
INPUT_SIZE    = GLOBAL_BINS + LOCAL_BINS + 2   # 264
BLS_PERIODS   = np.linspace(0.5, 25, 500)
BLS_DURATIONS = np.linspace(0.01, 0.3, 10)
MAX_SECTORS   = 3
N_SCALARS     = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── CNN model (identical to train_classifier.py) ──────────────────────────────

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
            nn.Linear(64, 1),              nn.Sigmoid(),
        )

    def forward(self, x, s):
        conv_out   = self.conv(x).flatten(1)
        scalar_out = self.scalar_branch(s)
        merged     = torch.cat([conv_out, scalar_out], dim=1)
        return self.fc(merged).squeeze(1).clamp(1e-7, 1 - 1e-7)


# ── Light curve processing (identical to train_classifier.py) ─────────────────

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


def process_tic(tic_id: str) -> tuple[np.ndarray | None, dict]:
    """Download, process, and return (264-vector, bls_info) for a TIC ID."""
    try:
        import lightkurve as lk
        from astropy.timeseries import BoxLeastSquares

        all_times, all_fluxes = [], []

        res = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
        if res is None or len(res) == 0:
            print("No SPOC data found.")
            return None, {}

        sectors_found = list(res.table["sequence_number"])
        print(f"Found {len(sectors_found)} sector(s): {sectors_found[:MAX_SECTORS]}")

        for i in range(min(MAX_SECTORS, len(res))):
            try:
                lc = res[i].download(quality_bitmask="default")
                if lc is None:
                    continue
                lc = lc.remove_nans().remove_outliers(sigma=5).flatten(window_length=401).normalize()
                all_times.append(lc.time.value)
                all_fluxes.append(lc.flux.value)
                time.sleep(0.05)
            except Exception as e:
                print(f"  Sector {i+1} failed: {e}")
                continue

        if not all_times:
            print("Failed to download any sectors.")
            return None, {}

        times  = np.concatenate(all_times)
        fluxes = np.concatenate(all_fluxes)

        bls   = BoxLeastSquares(times, fluxes)
        power = bls.power(BLS_PERIODS, BLS_DURATIONS)

        best_idx      = np.argmax(power.power)
        best_period   = float(power.period[best_idx])
        best_t0       = float(power.transit_time[best_idx])
        best_duration = float(power.duration[best_idx])
        best_depth    = float(power.depth[best_idx])

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

        # Secondary eclipse depth
        sec_half = max(best_duration * 2.0, 0.02)
        sec_mask = (phase >= 0.5 - sec_half) & (phase <= 0.5 + sec_half)
        sec_depth = float(1.0 - np.median(fluxes[sec_mask])) if sec_mask.sum() > 0 else 0.0

        # Even/odd depth difference
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
        vec   = np.concatenate([global_view, local_view, extra]).astype(np.float32)

        bls_info = {
            "period":        best_period,
            "duration_hr":   best_duration * 24,
            "depth_ppm":     best_depth * 1e6,
            "t0":            best_t0,
            "sec_depth":     sec_depth,
            "even_odd_diff": even_odd_diff,
            "sectors_used":  len(all_times),
        }
        return vec, bls_info

    except Exception as e:
        print(f"Error: {e}")
        return None, {}


# ── Scalar normalization (approximate — uses typical TESS dataset stats) ───────

# These are approximate log-scale means/stds derived from the TESS training set.
# If you have the labeled_tess_dataset.csv handy, pass --csv to compute exactly.
SCALAR_MEAN = np.array([1.5, 8.5, 2.8], dtype=np.float32)  # log1p(period), log1p(depth_ppm), duration_hr
SCALAR_STD  = np.array([1.2, 2.1, 1.6], dtype=np.float32)


def normalize_scalars(period: float, depth_ppm: float, duration_hr: float,
                      csv_path: str | None = None) -> np.ndarray:
    """Normalize scalars the same way load_cache does during training."""
    raw = np.array([np.log1p(period), np.log1p(depth_ppm), duration_hr], dtype=np.float32)

    if csv_path and Path(csv_path).exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        periods   = np.log1p(df["period"].fillna(0).values.astype(float))
        depths    = np.log1p(df["depth_ppm"].fillna(0).values.astype(float))
        durations = df["duration_hr"].fillna(0).values.astype(float)
        all_s = np.stack([periods, depths, durations], axis=1).astype(np.float32)
        mean  = all_s.mean(axis=0)
        std   = all_s.std(axis=0) + 1e-8
    else:
        mean = SCALAR_MEAN
        std  = SCALAR_STD

    return (raw - mean) / std


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CNN exoplanet inference on a TESS TIC target.")
    parser.add_argument("tic_id",          type=str,   help="TIC ID to classify")
    parser.add_argument("--model",         type=str,   default="exoplanet_cnn.pt",
                        help="Path to saved model weights (default: exoplanet_cnn.pt)")
    parser.add_argument("--csv",           type=str,   default="labeled_tess_dataset.csv",
                        help="Labeled TESS CSV for accurate scalar normalization")
    parser.add_argument("--period",        type=float, default=None,
                        help="Override BLS period (days) — skips download if all 3 provided")
    parser.add_argument("--depth",         type=float, default=None,
                        help="Override transit depth (ppm)")
    parser.add_argument("--duration",      type=float, default=None,
                        help="Override transit duration (hours)")
    args = parser.parse_args()

    print(f"\nDevice: {DEVICE}")

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found at '{model_path}'. Run train_classifier.py first.")
        return

    model = ExoplanetCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model: {model_path}\n")

    # Process light curve
    print(f"Processing TIC {args.tic_id}...")
    vec, bls_info = process_tic(args.tic_id)

    if vec is None:
        print("Could not process light curve. Exiting.")
        return

    # Use BLS-derived scalars unless overridden
    period     = args.period   if args.period   is not None else bls_info["period"]
    depth_ppm  = args.depth    if args.depth    is not None else bls_info["depth_ppm"]
    duration_hr = args.duration if args.duration is not None else bls_info["duration_hr"]

    scalars = normalize_scalars(period, depth_ppm, duration_hr, csv_path=args.csv)

    # Run inference
    X = torch.tensor(vec).unsqueeze(0).unsqueeze(0).to(DEVICE)   # (1, 1, 264)
    S = torch.tensor(scalars).unsqueeze(0).to(DEVICE)             # (1, 3)

    with torch.no_grad():
        prob = float(model(X, S).cpu().item())

    label = "✅  Planet candidate" if prob >= 0.5 else "❌  False positive"

    print(f"\n── BLS Results ──────────────────────────────────────")
    print(f"  Period          : {bls_info['period']:.4f} days")
    print(f"  Duration        : {bls_info['duration_hr']:.2f} hours")
    print(f"  Depth           : {bls_info['depth_ppm']:.0f} ppm")
    print(f"  Sec. eclipse    : {bls_info['sec_depth']:.6f}  (high = likely EB)")
    print(f"  Even/odd diff   : {bls_info['even_odd_diff']:.6f}  (high = likely EB)")
    print(f"  Sectors used    : {bls_info['sectors_used']}")
    print(f"\n── CNN Result ───────────────────────────────────────")
    print(f"  {label}")
    print(f"  Planet probability : {prob:.2%}")
    print(f"  FP probability     : {1-prob:.2%}")
    print(f"────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
