import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
import joblib
import os
import time

CLASSIFIER_PATH = "exoplanet_rf_classifier.joblib"

# Fine grid for inference — accurate single-target results
BLS_PERIODS   = np.linspace(0.5, 20, 10000)
BLS_DURATIONS = np.linspace(0.02, 0.3, 20)

# Feature order must exactly match train_classifier.py
FEATURE_NAMES = [
    "bls_period", "bls_duration", "bls_depth", "bls_snr", "bls_period_std",
    "depth_over_duration",
    "toi_period", "toi_depth", "toi_duration", "n_sectors",
]


def run_bls_on_lc(lc):
    """Run BLS on a single normalised light curve, return feature dict."""
    lc_clean = lc.remove_nans().remove_outliers(sigma=5).flatten(window_length=401).normalize()

    bls   = BoxLeastSquares(lc_clean.time.value, lc_clean.flux.value)
    power = bls.power(BLS_PERIODS, BLS_DURATIONS)

    best_index    = np.argmax(power.power)
    best_period   = float(power.period[best_index])
    best_duration = float(power.duration[best_index])
    best_t0       = float(power.transit_time[best_index])
    best_depth    = float(power.depth[best_index])
    bls_snr       = float(power.power[best_index] / (np.median(power.power) + 1e-9))

    return {
        "period":   best_period,
        "duration": best_duration,
        "depth":    best_depth,
        "t0":       best_t0,
        "snr":      bls_snr,
        "lc_clean": lc_clean,
    }


def searchSector(tic_id: str):
    print(f"\nSearching TIC {tic_id}...")

    search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
    if search is None or len(search) == 0:
        print("No SPOC data. Trying any pipeline...")
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
    if search is None or len(search) == 0:
        print(f"ERROR: No light curves found for TIC {tic_id}.")
        return

    n_sectors = len(search)
    print(f"Found {n_sectors} sector(s). Running BLS on up to 3...\n")

    # ── Run BLS on up to 3 sectors and average (mirrors train_classifier.py) ──
    sector_results = []
    for i in range(min(3, n_sectors)):
        sector_num = int(search.table["sequence_number"][i])
        print(f"  Sector {sector_num}...")
        try:
            lc = search[i].download(quality_bitmask="default")
            if lc is None:
                continue
            r = run_bls_on_lc(lc)
            sector_results.append(r)
            time.sleep(0.1)
        except Exception as e:
            print(f"    Skipped sector {sector_num}: {e}")

    if not sector_results:
        print("ERROR: BLS failed on all sectors.")
        return

    # Average across sectors
    best_period   = float(np.mean([r["period"]   for r in sector_results]))
    best_duration = float(np.mean([r["duration"] for r in sector_results]))
    best_depth    = float(np.mean([r["depth"]    for r in sector_results]))
    bls_snr       = float(np.mean([r["snr"]      for r in sector_results]))
    period_std    = float(np.std( [r["period"]   for r in sector_results])) if len(sector_results) > 1 else 0.0
    best_t0       = float(sector_results[0]["t0"])
    depth_over_dur = best_depth / (best_duration + 1e-6)

    print(f"\n── BLS Results ({len(sector_results)} sector(s) averaged) ────────────────")
    print(f"  Orbital period     : {best_period:.4f} days")
    print(f"  Period std         : {period_std:.4f} days  (low = stable signal)")
    print(f"  Transit duration   : {best_duration:.4f} days")
    print(f"  Fractional depth   : {best_depth * 100:.4f}%")
    print(f"  BLS SNR (proxy)    : {bls_snr:.2f}")
    print(f"  Depth/duration     : {depth_over_dur:.4f}")

    # ── Fold and plot using first sector's cleaned LC ─────────────────────────
    lc_folded = sector_results[0]["lc_clean"].fold(period=best_period, epoch_time=best_t0)
    lc_folded.plot(title=f"TIC {tic_id}  |  P={best_period:.3f}d  depth={best_depth*100:.3f}%  SNR={bls_snr:.1f}")

    # ── Classify ──────────────────────────────────────────────────────────────
    if not os.path.exists(CLASSIFIER_PATH):
        print(f"\n[WARNING] Classifier not found at '{CLASSIFIER_PATH}'.")
        print("Run train_classifier.py first.")
        plt.show()
        return

    clf = joblib.load(CLASSIFIER_PATH)

    # Feature order must match FEATURE_NAMES / train_classifier.py exactly
    features = np.array([[
        best_period,       # bls_period
        best_duration,     # bls_duration
        best_depth,        # bls_depth
        bls_snr,           # bls_snr
        period_std,        # bls_period_std
        depth_over_dur,    # depth_over_duration
        best_period,       # toi_period  (proxy — no TOI table at inference time)
        best_depth,        # toi_depth   (proxy)
        best_duration,     # toi_duration (proxy)
        float(n_sectors),  # n_sectors
    ]])

    prediction  = clf.predict(features)[0]
    probability = clf.predict_proba(features)[0]

    label_map = {0: "❌  Not a planet (False Positive)", 1: "✅  Planet candidate"}
    print(f"\n── Classifier Result ─────────────────────────────────")
    print(f"  Prediction  : {label_map[prediction]}")
    print(f"  Confidence  : planet={probability[1]:.2%}  |  non-planet={probability[0]:.2%}")
    print(f"──────────────────────────────────────────────────────")

    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
# Change TIC_ID to test any target
TIC_ID = "307210830"

searchSector(TIC_ID)
