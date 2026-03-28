"""
predict.py
──────────
Classify a single TESS target from the command line.

Usage:
    python predict.py <TIC_ID>
    python predict.py <TIC_ID> --sector <N>
    python predict.py <TIC_ID> --no-plot

Examples:
    python predict.py 307210830
    python predict.py 381472147 --sector 14
    python predict.py 307210830 --no-plot
"""

import argparse
import os
import sys
import time

import joblib
import numpy as np
from astropy.timeseries import BoxLeastSquares

CLASSIFIER_PATH = "exoplanet_rf_classifier.joblib"

BLS_PERIODS   = np.linspace(0.5, 20, 10000)
BLS_DURATIONS = np.linspace(0.02, 0.3, 20)


def run_bls_on_lc(lc):
    lc_clean = lc.remove_nans().remove_outliers(sigma=5).flatten(window_length=401).normalize()
    bls      = BoxLeastSquares(lc_clean.time.value, lc_clean.flux.value)
    power    = bls.power(BLS_PERIODS, BLS_DURATIONS)

    best_index    = np.argmax(power.power)
    return {
        "period":   float(power.period[best_index]),
        "duration": float(power.duration[best_index]),
        "depth":    float(power.depth[best_index]),
        "t0":       float(power.transit_time[best_index]),
        "snr":      float(power.power[best_index] / (np.median(power.power) + 1e-9)),
        "lc_clean": lc_clean,
    }


def predict(tic_id: str, sector: int = None, plot: bool = True):
    import lightkurve as lk

    # ── Search ────────────────────────────────────────────────────────────────
    print(f"\nTIC {tic_id}" + (f"  sector {sector}" if sector else "") + " — fetching...")
    search = lk.search_lightcurve(
        f"TIC {tic_id}", mission="TESS", author="SPOC",
        **({"sector": sector} if sector else {})
    )
    if search is None or len(search) == 0:
        print("No SPOC data found. Trying any pipeline...")
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS",
                                      **({"sector": sector} if sector else {}))
    if search is None or len(search) == 0:
        print(f"ERROR: No light curves found for TIC {tic_id}.")
        sys.exit(1)

    n_sectors     = len(search)
    sectors_found = list(search.table["sequence_number"])
    print(f"Found {n_sectors} sector(s): {sectors_found}")

    # ── BLS across up to 3 sectors ────────────────────────────────────────────
    results = []
    for i in range(min(3, n_sectors)):
        s = int(search.table["sequence_number"][i])
        print(f"  Running BLS on sector {s}...")
        try:
            lc = search[i].download(quality_bitmask="default")
            if lc:
                results.append(run_bls_on_lc(lc))
            time.sleep(0.1)
        except Exception as e:
            print(f"  Skipped sector {s}: {e}")

    if not results:
        print("ERROR: BLS failed on all sectors.")
        sys.exit(1)

    # ── Average features ──────────────────────────────────────────────────────
    period    = float(np.mean([r["period"]   for r in results]))
    duration  = float(np.mean([r["duration"] for r in results]))
    depth     = float(np.mean([r["depth"]    for r in results]))
    snr       = float(np.mean([r["snr"]      for r in results]))
    period_std = float(np.std([r["period"]   for r in results])) if len(results) > 1 else 0.0
    t0        = results[0]["t0"]
    dod       = depth / (duration + 1e-6)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'─'*52}")
    print(f"  TIC ID             : {tic_id}")
    print(f"  Sectors used       : {[int(search.table['sequence_number'][i]) for i in range(min(3, n_sectors))]}")
    print(f"  Orbital period     : {period:.4f} days")
    print(f"  Period stability   : ±{period_std:.4f} days  {'✅ stable' if period_std < 0.5 else '⚠️  variable'}")
    print(f"  Transit duration   : {duration:.4f} days  ({duration*24:.2f} hours)")
    print(f"  Transit depth      : {depth*100:.4f}%  ({depth*1e6:.1f} ppm)")
    print(f"  First transit (t0) : {t0:.4f} BTJD")
    print(f"  BLS SNR            : {snr:.2f}  {'✅ strong' if snr > 8 else '⚠️  weak'}")
    print(f"  Depth/duration     : {dod:.4f}")
    print(f"{'─'*52}")

    # ── Classify ──────────────────────────────────────────────────────────────
    if not os.path.exists(CLASSIFIER_PATH):
        print(f"\n[WARNING] No model found at '{CLASSIFIER_PATH}'. Run train_classifier.py first.")
    else:
        clf = joblib.load(CLASSIFIER_PATH)
        features = np.array([[
            period, duration, depth, snr, period_std, dod,
            period, depth, duration,   # toi proxies
            float(n_sectors),
        ]])
        pred  = clf.predict(features)[0]
        proba = clf.predict_proba(features)[0]

        label = "✅  Planet candidate" if pred == 1 else "❌  False positive"
        print(f"\n  Prediction  : {label}")
        print(f"  Confidence  : planet={proba[1]:.2%}  |  non-planet={proba[0]:.2%}")
        print(f"{'─'*52}\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if plot:
        import matplotlib.pyplot as plt
        lc_folded = results[0]["lc_clean"].fold(period=period, epoch_time=t0)
        lc_folded.plot(title=f"TIC {tic_id}  |  P={period:.3f}d  depth={depth*100:.3f}%  SNR={snr:.1f}")
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Classify a TESS target as planet candidate or false positive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict.py 307210830\n"
            "  python predict.py 381472147 --sector 14\n"
            "  python predict.py 307210830 --no-plot\n"
        )
    )
    parser.add_argument("tic_id",          type=str,            help="TIC ID to classify")
    parser.add_argument("--sector", "-s",  type=int, default=None, help="Specific sector to use (default: first available)")
    parser.add_argument("--no-plot",       action="store_true", help="Skip the folded light curve plot")
    args = parser.parse_args()

    predict(
        tic_id  = args.tic_id,
        sector  = args.sector,
        plot    = not args.no_plot,
    )


if __name__ == "__main__":
    main()
