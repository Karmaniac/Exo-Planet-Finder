import argparse
import json
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

def _pick_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        nn.Conv1d(1, 1, 3).to("cuda")(torch.zeros(1, 1, 8, device="cuda"))
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")   # cuda present but engine broken (driver/cuDNN mismatch)


DEVICE = _pick_device()


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
            nn.Linear(64, 1),
        )

    def forward(self, x, s):
        conv_out   = self.conv(x).flatten(1)
        scalar_out = self.scalar_branch(s)
        merged     = torch.cat([conv_out, scalar_out], dim=1)
        return torch.sigmoid(self.fc(merged).squeeze(1)).clamp(1e-7, 1 - 1e-7)


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
# If you have the data/labeled_tess_dataset.csv handy, pass --csv to compute exactly.
# Approximate fallback stats — pass --csv for exact normalization
SCALAR_MEAN = np.array([1.5, 8.5, 2.8], dtype=np.float32)
SCALAR_STD  = np.array([1.2, 2.1, 1.6], dtype=np.float32)


def normalize_scalars(period: float, depth_ppm: float, duration_hr: float,
                      csv_path: str | None = None) -> np.ndarray:
    """Normalize scalars exactly as load_cache does during training."""
    raw = np.array([np.log1p(period), np.log1p(depth_ppm), duration_hr], dtype=np.float32)

    if csv_path and Path(csv_path).exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        periods   = np.log1p(df["period"].fillna(0).values.astype(float))
        depths    = np.log1p(df["depth_ppm"].fillna(0).values.astype(float))
        durations = df["duration_hr"].fillna(0).values.astype(float)
        all_s = np.stack([periods, depths, durations], axis=1).astype(np.float32)
        # Match load_cache: nan_to_num then normalize
        all_s = np.nan_to_num(all_s, nan=0.0, posinf=0.0, neginf=0.0)
        mean  = all_s.mean(axis=0)
        std   = all_s.std(axis=0) + 1e-8
    else:
        mean = SCALAR_MEAN
        std  = SCALAR_STD

    return np.nan_to_num((raw - mean) / std, nan=0.0)


# ── Shared prediction logic (used by CLI and the GUI) ──────────────────────────

def predict_tic(tic_id: str, model: "ExoplanetCNN", csv_path: str = "data/labeled_tess_dataset.csv",
                cache_dir: str = "lc_cache", force_download: bool = False,
                period: float = None, depth: float = None, duration: float = None) -> tuple[float, dict]:
    """Run the full feature pipeline + model for a TIC ID. Returns (probability, bls_info)."""
    import pandas as pd

    npy_path = Path(cache_dir) / f"{tic_id}.npy"

    if npy_path.exists() and not force_download:
        vec = np.load(npy_path)

        p, d, dur = 0.0, 0.0, 0.0
        if csv_path and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            df["TIC_ID"] = df["TIC_ID"].astype(str)
            row = df[df["TIC_ID"] == str(tic_id)]
            if not row.empty:
                p   = float(row.iloc[0].get("period",      0.0) or 0.0)
                d   = float(row.iloc[0].get("depth_ppm",   0.0) or 0.0)
                dur = float(row.iloc[0].get("duration_hr", 0.0) or 0.0)

        bls_info = {
            "period": period if period is not None else p,
            "depth_ppm": depth if depth is not None else d,
            "duration_hr": duration if duration is not None else dur,
            "sec_depth": float(vec[-2]), "even_odd_diff": float(vec[-1]),
            "sectors_used": "cached",
        }
    else:
        vec, bls_info = process_tic(tic_id)
        if vec is None:
            return None, {}
        bls_info["period"]      = period if period is not None else bls_info["period"]
        bls_info["depth_ppm"]   = depth if depth is not None else bls_info["depth_ppm"]
        bls_info["duration_hr"] = duration if duration is not None else bls_info["duration_hr"]

    scalars = normalize_scalars(bls_info["period"], bls_info["depth_ppm"], bls_info["duration_hr"], csv_path=csv_path)

    X = torch.tensor(vec).unsqueeze(0).unsqueeze(0).to(DEVICE)
    S = torch.tensor(scalars).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = float(model(X, S).cpu().item())

    return prob, bls_info, vec


# ── Stellar parameters + physical planet properties ────────────────────────────

STELLAR_CACHE_PATH = "stellar_cache.json"

# Unit conversions
RSUN_REARTH = 109.2     # solar radii -> Earth radii
RSUN_AU     = 0.00465047  # solar radii -> AU
REARTH_RJUP = 1 / 11.2
ALBEDO      = 0.3       # assumed Bond albedo for Teq


def fetch_stellar_params(tic_id: str, cache_path: str = STELLAR_CACHE_PATH) -> dict | None:
    """Look up stellar radius/mass/Teff for a TIC ID from the MAST TIC catalog. Disk-cached."""
    cache_file = Path(cache_path)
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    if str(tic_id) in cache:
        return cache[str(tic_id)]

    try:
        from astroquery.mast import Catalogs
        table = Catalogs.query_criteria(catalog="Tic", ID=str(tic_id))
        if len(table) == 0:
            return None
        row = table[0]
        params = {
            "rad_rsun":  float(row["rad"]),
            "mass_msun": float(row["mass"]),
            "teff_k":    float(row["Teff"]),
        }
        if any(np.isnan(v) for v in params.values()):
            return None
    except Exception:
        return None

    cache[str(tic_id)] = params
    cache_file.write_text(json.dumps(cache, indent=2))
    return params


def planet_physical_params(bls_info: dict, stellar: dict) -> dict | None:
    """Combine BLS transit geometry with stellar params into physical planet properties."""
    depth = bls_info["depth_ppm"] / 1e6
    period_days = bls_info["period"]
    rstar_rsun = stellar["rad_rsun"]
    mstar_msun = stellar["mass_msun"]
    teff_k = stellar["teff_k"]

    if depth <= 0 or period_days <= 0 or rstar_rsun <= 0 or mstar_msun <= 0:
        return None

    rp_rsun = (depth ** 0.5) * rstar_rsun
    rp_earth = rp_rsun * RSUN_REARTH

    a_au = (mstar_msun * (period_days / 365.25) ** 2) ** (1 / 3)
    teq_k = teff_k * ((rstar_rsun * RSUN_AU) / (2 * a_au)) ** 0.5 * (1 - ALBEDO) ** 0.25
    insolation_earth = (rstar_rsun ** 2) * ((teff_k / 5772.0) ** 4) / (a_au ** 2)

    mass_earth = estimate_mass_earth(rp_earth)
    density_gcm3 = (
        EARTH_DENSITY_GCM3 * mass_earth / (rp_earth ** 3) if mass_earth is not None else None
    )

    return {
        "rp_earth": rp_earth,
        "rp_jupiter": rp_earth * REARTH_RJUP,
        "a_au": a_au,
        "teq_k": teq_k,
        "insolation_earth": insolation_earth,
        "mass_earth": mass_earth,
        "density_gcm3": density_gcm3,
        "type_label": classify_planet(rp_earth, teq_k),
    }


# ── Mass / density / type estimate (radius-only, no stellar RV needed) ─────────
#
# Inverted from Chen & Kipping (2017) forecaster power laws (mass -> radius),
# solved for mass given radius. Approximate — real mass needs RV or TTVs.

NEPTUNIAN_R_MIN = 1.23   # R_earth, Terran/Neptunian boundary
JOVIAN_R_MIN    = 14.3   # R_earth, Neptunian/Jovian boundary (mass ill-constrained beyond this)
EARTH_DENSITY_GCM3 = 5.51


def estimate_mass_earth(rp_earth: float) -> float | None:
    if rp_earth < NEPTUNIAN_R_MIN:
        return (rp_earth / 1.008) ** (1 / 0.279)
    if rp_earth < JOVIAN_R_MIN:
        return (rp_earth / 0.808) ** (1 / 0.589)
    return None   # gas giant regime: radius barely depends on mass, estimate unreliable


def classify_planet(rp_earth: float, teq_k: float) -> str:
    if rp_earth < 0.8:
        size = "Sub-Earth"
    elif rp_earth < 1.25:
        size = "Earth-like"
    elif rp_earth < 2.0:
        size = "Super-Earth"
    elif rp_earth < 4.0:
        size = "Sub-Neptune"
    elif rp_earth < 10.0:
        size = "Neptune-like"
    else:
        size = "Jovian"

    if teq_k >= 1000:
        return f"Hot {size}"
    if teq_k < 200:
        return f"Cold {size}"
    return size


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CNN exoplanet inference on a TESS TIC target.")
    parser.add_argument("tic_id",          type=str,   help="TIC ID to classify")
    parser.add_argument("--model",         type=str,   default="models/exoplanet_cnn.pt",
                        help="Path to saved model weights (default: models/exoplanet_cnn.pt)")
    parser.add_argument("--csv",           type=str,   default="data/labeled_tess_dataset.csv",
                        help="Labeled TESS CSV for accurate scalar normalization")
    parser.add_argument("--period",        type=float, default=None,
                        help="Override BLS period (days) — skips download if all 3 provided")
    parser.add_argument("--depth",         type=float, default=None,
                        help="Override transit depth (ppm)")
    parser.add_argument("--duration",      type=float, default=None,
                        help="Override transit duration (hours)")
    parser.add_argument("--cache",         type=str,   default="lc_cache",
                        help="Cache directory to check before downloading (default: lc_cache)")
    parser.add_argument("--force-download", action="store_true",
                        help="Force fresh download even if cached")
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

    prob, bls_info, _vec = predict_tic(
        args.tic_id, model, csv_path=args.csv, cache_dir=args.cache,
        force_download=args.force_download,
        period=args.period, depth=args.depth, duration=args.duration,
    )
    if prob is None:
        print("Could not process light curve. Exiting.")
        return

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