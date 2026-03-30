"""
fetch_astronet_dataset.py
─────────────────────────
Downloads the Astronet-Triage tces.csv from GitHub and converts it into
the labeled_tess_dataset.csv format used by train_classifier.py download.

Labels:
  PC (planet candidate) → 1
  EB, IS, V, O, J       → 0  (all false positive / non-planet classes)

Output:
  astronet_tess_dataset.csv — ready to feed into train_classifier.py download

Usage:
  python fetch_astronet_dataset.py
  python fetch_astronet_dataset.py --exclude-existing labeled_tess_dataset.csv
  python fetch_astronet_dataset.py --max-targets 2000 --exclude-existing labeled_tess_dataset.csv
"""

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

TCES_URL = "https://raw.githubusercontent.com/yuliang419/Astronet-Triage/master/astronet/tces.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",              type=str, default="astronet_tess_dataset.csv",
                        help="Output CSV path (default: astronet_tess_dataset.csv)")
    parser.add_argument("--exclude-existing", type=str, default=None,
                        help="Existing labeled CSV — TIC IDs already in there will be excluded")
    parser.add_argument("--max-targets",      type=int, default=None,
                        help="Cap total targets (stratified by label)")
    args = parser.parse_args()

    # ── Download tces.csv ─────────────────────────────────────────────────────
    print("Downloading Astronet-Triage tces.csv...")
    resp = requests.get(TCES_URL, timeout=30)
    resp.raise_for_status()
    raw = pd.read_csv(StringIO(resp.text))
    print(f"Downloaded {len(raw):,} rows.")

    # Normalize column names
    raw.columns = raw.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── Find key columns ──────────────────────────────────────────────────────
    tic_col  = next((c for c in raw.columns if "tic" in c), None)
    disp_col = next((c for c in raw.columns if "disp" in c), None)

    if tic_col is None:
        raise ValueError(f"Could not find TIC ID column. Columns: {raw.columns.tolist()}")
    if disp_col is None:
        raise ValueError(f"Could not find disposition column. Columns: {raw.columns.tolist()}")

    print(f"TIC column: '{tic_col}'  |  Disposition column: '{disp_col}'")
    print(f"\nDisposition breakdown:\n{raw[disp_col].value_counts().to_string()}\n")

    # ── Filter to only PC and EB ─────────────────────────────────────────────
    # Keep only planet candidates and eclipsing binaries — the hard problem.
    # IS, V, O, J (instrumental/variability/junk) are easy to reject and
    # don't help the model learn to distinguish planets from EBs.
    raw["disp_clean"] = raw[disp_col].str.strip().str.upper()
    before = len(raw)
    raw = raw[raw["disp_clean"].isin(["PC", "EB"])].reset_index(drop=True)
    print(f"Kept PC + EB only: {len(raw):,} rows (dropped {before - len(raw):,} IS/V/O/J rows)")

    # ── Map labels ────────────────────────────────────────────────────────────
    # PC = planet candidate → 1
    # EB = eclipsing binary → 0
    raw["label"] = raw["disp_clean"].apply(
        lambda d: 1 if d == "PC" else 0
    )

    # ── Find optional scalar columns ──────────────────────────────────────────
    period_col   = next((c for c in raw.columns if "period" in c), None)
    duration_col = next((c for c in raw.columns if "duration" in c), None)
    depth_col    = next((c for c in raw.columns if "depth" in c), None)
    sector_col   = next((c for c in raw.columns if "sector" in c and "n_" not in c), None)

    # ── Build output dataframe ────────────────────────────────────────────────
    out = pd.DataFrame()
    out["TIC_ID"] = raw[tic_col].astype(str).str.strip()
    out["label"]  = raw["label"]

    if period_col:
        out["period"] = pd.to_numeric(raw[period_col], errors="coerce").fillna(0.0)
    else:
        out["period"] = 0.0

    if depth_col:
        depth_vals = pd.to_numeric(raw[depth_col], errors="coerce").fillna(0.0)
        # Convert fractional depth to ppm if values look like fractions (< 1)
        out["depth_ppm"] = depth_vals.apply(lambda d: d * 1e6 if 0 < d < 1.0 else d)
    else:
        out["depth_ppm"] = 0.0

    if duration_col:
        dur_vals = pd.to_numeric(raw[duration_col], errors="coerce").fillna(0.0)
        # Convert days to hours if values look like days (< 2.0)
        out["duration_hr"] = dur_vals.apply(lambda d: d * 24 if 0 < d < 2.0 else d)
    else:
        out["duration_hr"] = 0.0

    if sector_col:
        out["sectors"]   = raw[sector_col].apply(
            lambda s: f"[{int(s)}]" if pd.notna(s) else "[]"
        )
        out["n_sectors"] = 1
    else:
        out["sectors"]   = "[]"
        out["n_sectors"] = 0

    # ── Drop duplicates ───────────────────────────────────────────────────────
    before = len(out)
    out = out.drop_duplicates(subset="TIC_ID", keep="first").reset_index(drop=True)
    print(f"Dropped {before - len(out):,} duplicate TIC IDs.")

    # ── Exclude TIC IDs already in existing dataset ───────────────────────────
    if args.exclude_existing and Path(args.exclude_existing).exists():
        existing    = pd.read_csv(args.exclude_existing)
        existing_ids = set(existing["TIC_ID"].astype(str))
        before = len(out)
        out = out[~out["TIC_ID"].isin(existing_ids)].reset_index(drop=True)
        print(f"Excluded {before - len(out):,} TIC IDs already in '{args.exclude_existing}'.")

    # ── Optional cap ──────────────────────────────────────────────────────────
    if args.max_targets:
        out = out.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.max_targets // 2), random_state=42)
        ).reset_index(drop=True)
        print(f"Capped to {len(out):,} targets (stratified).")

    # ── Summary ───────────────────────────────────────────────────────────────
    counts = out["label"].value_counts()
    print(f"\nFinal dataset: {len(out):,} targets")
    print(f"  Planets (PC)       : {counts.get(1, 0):,}")
    print(f"  False positives    : {counts.get(0, 0):,}")

    out.to_csv(args.out, index=False)
    print(f"\nSaved to: {args.out}")
    print(f"\nNext step — download light curves:")
    print(f"  python train_classifier.py download --csv {args.out} --cache lc_cache")


if __name__ == "__main__":
    main()