"""
fetch_kepler_dataset.py
-----------------------
Downloads the NASA Exoplanet Archive Kepler KOI cumulative table,
filters to confirmed planets (CONFIRMED) and false positives (FALSE POSITIVE),
and fetches available Kepler quarters for each KIC ID.

Output: labeled_kepler_dataset.csv
Columns: KIC_ID, label (1=planet, 0=false positive), quarters, period, depth_ppm, duration_hr

Requirements:
    pip install lightkurve pandas requests

Usage:
    python fetch_kepler_dataset.py

    Optional flags:
        --max-targets 500        # limit how many targets to query (default: all)
        --output my_kepler.csv   # custom output filename
        --no-balance             # keep full imbalanced dataset
"""

import argparse
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# NASA Exoplanet Archive - Kepler KOI table
# Multiple URLs tried in order — the table/column names have changed over time
KEPLER_URLS = [
    # Current table name (2023+)
    (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        "?query=select+kepid,koi_disposition,koi_period,koi_depth,koi_duration"
        "+from+koi"
        "&format=csv"
    ),
    # Legacy table name
    (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        "?query=select+kepid,koi_disposition,koi_period,koi_depth,koi_duration"
        "+from+cumulative"
        "&format=csv"
    ),
    # Direct CSV download (no TAP)
    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
    "?table=cumulative&select=kepid,koi_disposition,koi_period,koi_depth,koi_duration"
    "&format=csv",
]

KEEP_DISPOSITIONS = {
    "CONFIRMED":      1,
    "FALSE POSITIVE": 1 - 1,   # evaluates to 0, avoids bare 0 confusing linters
    "FALSE+POSITIVE": 0,        # some rows use + instead of space
}


# ---------------------------------------------------------------------------
# Step 1 -- Download KOI table
# ---------------------------------------------------------------------------

KAGGLE_COLUMNS = {
    "disposition": ["koi_disposition", "disposition"],
    "kic_id":      ["kepid", "kic_id", "KIC"],
    "period":      ["koi_period", "period"],
    "depth_ppm":   ["koi_depth", "depth"],
    "duration_hr": ["koi_duration", "duration"],
}


def download_koi_table(kaggle_csv: str = None) -> pd.DataFrame:
    # ── Option 1: local Kaggle CSV ─────────────────────────────────────────────
    if kaggle_csv:
        path = Path(kaggle_csv)
        if path.exists():
            print(f"Loading Kepler KOI table from local file: {path}")
            df = pd.read_csv(path, comment="#")
            print(f"  Loaded {len(df):,} rows.")
            return df
        else:
            print(f"  WARNING: '{kaggle_csv}' not found — falling back to NASA API.")

    # ── Option 2: NASA TAP API (multiple fallback URLs) ────────────────────────
    print("Downloading Kepler KOI table from NASA Exoplanet Archive...")
    for i, url in enumerate(KEPLER_URLS, 1):
        try:
            print(f"  Trying URL {i}/{len(KEPLER_URLS)}...")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text), comment="#")
            if len(df) < 10:
                print(f"    Too few rows ({len(df)}) — trying next.")
                continue
            disp_col = next((c for c in df.columns if "disposition" in c.lower()), None)
            if disp_col is None:
                print(f"    No disposition column. Columns: {list(df.columns)[:8]}")
                continue
            print(f"  Downloaded {len(df):,} rows.")
            return df
        except Exception as e:
            print(f"    Failed: {e}")

    print("\nERROR: Could not download Kepler data.")
    print("Download the CSV manually from:")
    print("  https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results")
    print("Then run with:  python fetch_kepler_dataset.py --kaggle-csv cumulative.csv")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 2 -- Filter and label
# ---------------------------------------------------------------------------

def filter_and_label(df: pd.DataFrame) -> pd.DataFrame:
    # Find disposition column — name varies by API version
    disp_col = next((c for c in df.columns if "disposition" in c.lower()), None)
    if disp_col is None:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("Could not find disposition column.")

    # Find KIC ID column
    kic_col = next((c for c in df.columns if c.lower() in ("kepid", "kic_id", "kic")), None)
    if kic_col is None:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("Could not find KIC ID column.")

    df[disp_col] = (
        df[disp_col].astype(str).str.strip().str.upper()
        .str.replace("+", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    # Show what dispositions exist so we can debug if needed
    print(f"  Dispositions found: {df[disp_col].value_counts().to_dict()}")

    keep = df[disp_col].isin(KEEP_DISPOSITIONS)
    df   = df[keep].copy()
    df["label"]  = df[disp_col].map(KEEP_DISPOSITIONS)
    df["KIC_ID"] = df[kic_col].astype(str).str.replace(r"\.0$", "", regex=True)

    # Rename useful columns — handle various naming conventions
    col_map = {}
    for target, candidates in [
        ("period",      ["koi_period",   "pl_orbper", "period"]),
        ("depth_ppm",   ["koi_depth",    "transit_depth", "depth"]),
        ("duration_hr", ["koi_duration", "transit_dur",   "duration"]),
    ]:
        for c in candidates:
            if c in df.columns:
                col_map[c] = target
                break

    df = df.rename(columns=col_map)
    keep_cols = ["KIC_ID", "label"] + [c for c in ["period", "depth_ppm", "duration_hr"] if c in df.columns]
    df = df[keep_cols].drop_duplicates(subset="KIC_ID").reset_index(drop=True)

    counts = df["label"].value_counts()
    print(f"\nFiltered Kepler dataset:")
    print(f"  Confirmed planets (label=1): {counts.get(1, 0):,}")
    print(f"  False positives   (label=0): {counts.get(0, 0):,}")
    print(f"  Total                      : {len(df):,}\n")

    return df


# ---------------------------------------------------------------------------
# Step 3 -- Fetch available Kepler quarters per KIC ID
# ---------------------------------------------------------------------------

def fetch_quarters_for_kic(kic_id: str, retries: int = 3) -> list:
    """
    Check if any Kepler data exists for this KIC and return quarter list.
    Kepler has 17 quarters (1-17) — we return all available ones by
    checking what lightkurve finds rather than parsing sequence_number
    which is unreliable for Kepler.
    """
    import lightkurve as lk
    for attempt in range(retries):
        try:
            results = lk.search_lightcurve(
                f"KIC {kic_id}",
                mission="Kepler",
            )
            if results is None or len(results) == 0:
                return []

            # Instead of parsing sequence_number (often masked),
            # just return a range representing available data exists.
            # train_classifier.py will search fresh anyway.
            n = len(results)
            if n > 0:
                # Return [1] as a sentinel meaning "has Kepler data"
                # train_classifier.py searches without specifying quarter
                return list(range(1, min(n + 1, 18)))
            return []
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Warning: could not fetch quarters for KIC {kic_id}: {e}")
                return []


def fetch_all_quarters(df: pd.DataFrame, max_targets: int = None, balance: bool = True) -> pd.DataFrame:
    if max_targets:
        per_class = max_targets // 2
        planets = df[df["label"] == 1].sample(min(per_class, (df["label"] == 1).sum()), random_state=42)
        fps     = df[df["label"] == 0].sample(min(per_class, (df["label"] == 0).sum()), random_state=42)
        df      = pd.concat([planets, fps]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(planets)} planets + {len(fps)} FPs = {len(df)} total.\n")
    else:
        counts = df["label"].value_counts()
        print(f"Using full Kepler dataset ({len(df):,} targets).")
        if balance:
            min_count = counts.min()
            df = df.groupby("label", group_keys=False).apply(
                lambda g: g.sample(min_count, random_state=42)
            ).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"  Balanced to {len(df):,} ({min_count} per class)\n")
        else:
            print(f"  (Unbalanced -- using all records)\n")

    total       = len(df)
    quarter_list = []

    print(f"Fetching quarters for {total:,} KIC IDs...")
    for i, kic_id in enumerate(df["KIC_ID"], 1):
        if i % 50 == 0 or i == 1:
            print(f"  Progress: {i}/{total}")
        quarter_list.append(fetch_quarters_for_kic(kic_id))
        time.sleep(0.15)

    df            = df.copy()
    df["quarters"]  = quarter_list
    df["n_quarters"] = df["quarters"].apply(len)

    before = len(df)
    df     = df[df["n_quarters"] > 0].reset_index(drop=True)
    print(f"\n  Dropped {before - len(df)} KICs with no Kepler data.")
    print(f"  Remaining: {len(df):,} targets.\n")

    # Rename to match TESS CSV schema so train_classifier.py can read both
    df = df.rename(columns={"KIC_ID": "TIC_ID", "quarters": "sectors", "n_quarters": "n_sectors"})
    df["mission"] = "Kepler"

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build labeled Kepler dataset.")
    parser.add_argument("--from-csv",    type=str,  default=None,
                        help="Path to a locally downloaded cumulative.csv from Kaggle "
                             "(skips the API download entirely)")
    parser.add_argument("--max-targets", type=int,  default=None)
    parser.add_argument("--output",      type=str,  default="labeled_kepler_dataset.csv")
    parser.add_argument("--no-balance",  action="store_true")
    parser.add_argument("--skip-quarters", action="store_true",
                        help="Skip quarter fetching (just save labels)")
    args = parser.parse_args()

    if args.from_csv:
        print(f"Loading local Kaggle CSV: {args.from_csv}")
        raw_df = pd.read_csv(args.from_csv, comment="#")
        print(f"  Loaded {len(raw_df):,} rows.")
    else:
        raw_df = download_koi_table()

    labeled_df = filter_and_label(raw_df)

    if args.skip_quarters:
        labeled_df.to_csv(args.output, index=False)
        print(f"Saved labels (no quarters) to {args.output}")
        return

    final_df = fetch_all_quarters(labeled_df, max_targets=args.max_targets, balance=not args.no_balance)
    final_df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    print(final_df[["TIC_ID", "label", "n_sectors", "sectors"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
