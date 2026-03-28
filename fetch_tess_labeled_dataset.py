import argparse
import ast
import sys
import time

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Step 1 — Download the TOI table from ExoFOP-TESS
# ---------------------------------------------------------------------------

TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"

# Fallback: NASA Exoplanet Archive TOI table (same data, different endpoint)
NASA_TOI_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+*+from+toi"
    "&format=csv"
)


def download_toi_table() -> pd.DataFrame:
    """Download the TOI table. Tries ExoFOP first, falls back to NASA Archive."""
    print("Downloading TOI table from ExoFOP-TESS...")
    try:
        r = requests.get(TOI_URL, timeout=60)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), comment="#")
        print(f"  Downloaded {len(df):,} rows from ExoFOP.")
        return df
    except Exception as e:
        print(f"  ExoFOP failed ({e}). Trying NASA Exoplanet Archive...")

    try:
        r = requests.get(NASA_TOI_URL, timeout=120)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), comment="#")
        print(f"  Downloaded {len(df):,} rows from NASA Archive.")
        return df
    except Exception as e:
        print(f"  NASA Archive also failed: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 2 — Filter to confirmed planets and false positives
# ---------------------------------------------------------------------------

# ExoFOP column name for disposition
DISP_COLS = ["TFOPWG Disposition", "tfopwg_disp", "disposition"]

# Labels we keep
KEEP_DISPOSITIONS = {
    "CP": 1,   # Confirmed Planet
    "KP": 1,   # Known Planet
    "FP": 0,   # False Positive
}


def filter_and_label(df: pd.DataFrame) -> pd.DataFrame:
    # Find the disposition column (name varies slightly by source)
    disp_col = None
    for col in DISP_COLS:
        if col in df.columns:
            disp_col = col
            break
    if disp_col is None:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("Could not find disposition column. Check column names above.")

    # Find TIC ID column
    tic_col = None
    for candidate in ["TIC ID", "tic_id", "TIC", "ticid"]:
        if candidate in df.columns:
            tic_col = candidate
            break
    if tic_col is None:
        raise ValueError(f"Could not find TIC ID column. Columns: {list(df.columns)}")

    df[disp_col] = df[disp_col].astype(str).str.strip().str.upper()
    df_filtered = df[df[disp_col].isin(KEEP_DISPOSITIONS)].copy()
    df_filtered["label"] = df_filtered[disp_col].map(KEEP_DISPOSITIONS)
    df_filtered["TIC_ID"] = df_filtered[tic_col].astype(str).str.replace(r"\.0$", "", regex=True)

    # Pull useful columns if present
    extra_cols = {}
    for alias, key in [
        (["Period (days)", "pl_orbper", "period"], "period"),
        (["Depth (ppm)", "transit_depth", "depth"], "depth_ppm"),
        (["Duration (hours)", "transit_dur", "duration"], "duration_hr"),
        (["TOI", "toi"], "TOI"),
    ]:
        for col in alias:
            if col in df.columns:
                extra_cols[key] = col
                break

    keep = ["TIC_ID", "label"] + list(extra_cols.values())
    result = df_filtered[[c for c in keep if c in df_filtered.columns]].copy()
    # Rename extra cols to clean names
    result = result.rename(columns={v: k for k, v in extra_cols.items()})

    counts = result["label"].value_counts()
    print(f"\nFiltered dataset:")
    print(f"  Confirmed / Known Planets (label=1): {counts.get(1, 0):,}")
    print(f"  False Positives          (label=0): {counts.get(0, 0):,}")
    print(f"  Total:                               {len(result):,}\n")

    return result.drop_duplicates(subset="TIC_ID")


# ---------------------------------------------------------------------------
# Step 3 — Fetch available TESS sectors per TIC ID via Lightkurve
# ---------------------------------------------------------------------------

def fetch_sectors_for_tic(tic_id: str, retries: int = 3) -> list:
    """Return a list of TESS sector numbers for a given TIC ID."""
    import lightkurve as lk
    for attempt in range(retries):
        try:
            results = lk.search_lightcurve(
                f"TIC {tic_id}",
                mission="TESS",
                author="SPOC",   # use SPOC pipeline (2-min cadence); change to "QLP" for FFI
            )
            if results is None or len(results) == 0:
                return []
            sectors = sorted(set(int(s) for s in results.table["sequence_number"] if s is not None))
            return sectors
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Warning: could not fetch sectors for TIC {tic_id}: {e}")
                return []


def fetch_all_sectors(df: pd.DataFrame, max_targets: int = None, balance: bool = True) -> pd.DataFrame:
    if max_targets:
        # Stratified subsample -- equal planets and FPs up to max_targets
        per_class = max_targets // 2
        planets = df[df["label"] == 1].sample(
            min(per_class, (df["label"] == 1).sum()), random_state=42
        )
        fps = df[df["label"] == 0].sample(
            min(per_class, (df["label"] == 0).sum()), random_state=42
        )
        df = pd.concat([planets, fps]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(planets)} planets + {len(fps)} false positives = {len(df)} total targets.\n")
    else:
        # Full dataset -- all confirmed planets and false positives
        counts = df["label"].value_counts()
        print(f"Using full dataset:")
        print(f"  Planets (1) : {counts.get(1, 0):,}")
        print(f"  FPs     (0) : {counts.get(0, 0):,}")
        print(f"  Total       : {len(df):,}")
        if balance:
            min_count = counts.min()
            df = df.groupby("label", group_keys=False).apply(
                lambda g: g.sample(min_count, random_state=42)
            ).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"  Balanced to : {len(df):,} ({min_count} per class)\n")
        else:
            print(f"  (Unbalanced -- using all records as-is)\n")

    total = len(df)
    sectors_list = []

    print(f"Fetching sectors for {total:,} TIC IDs (this may take a while)...")
    for i, tic_id in enumerate(df["TIC_ID"], 1):
        if i % 50 == 0 or i == 1:
            print(f"  Progress: {i}/{total}")
        sectors = fetch_sectors_for_tic(tic_id)
        sectors_list.append(sectors)
        time.sleep(0.15)   # be polite to MAST

    df = df.copy()
    df["sectors"] = sectors_list
    df["n_sectors"] = df["sectors"].apply(len)

    # Drop TICs with no data at all
    before = len(df)
    df = df[df["n_sectors"] > 0].reset_index(drop=True)
    print(f"\n  Dropped {before - len(df)} targets with no SPOC light curves.")
    print(f"  Remaining: {len(df):,} targets with at least one sector.\n")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a labeled TESS exoplanet dataset.")
    parser.add_argument("--max-targets", type=int, default=None,
                        help="Limit number of targets to query (default: all)")
    parser.add_argument("--output", type=str, default="labeled_tess_dataset.csv",
                        help="Output CSV filename (default: labeled_tess_dataset.csv)")
    parser.add_argument("--skip-sectors", action="store_true",
                        help="Skip sector fetching (just save labels from TOI table)")
    parser.add_argument("--no-balance", action="store_true",
                        help="When fetching all records, do NOT downsample majority class (default: balance)")
    args = parser.parse_args()

    # 1. Download TOI table
    raw_df = download_toi_table()

    # 2. Filter and label
    labeled_df = filter_and_label(raw_df)

    if args.skip_sectors:
        labeled_df.to_csv(args.output, index=False)
        print(f"Saved labels (no sectors) to {args.output}")
        return

    # 3. Fetch sectors
    final_df = fetch_all_sectors(labeled_df, max_targets=args.max_targets, balance=not args.no_balance)

    # 4. Save
    final_df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    print(final_df[["TIC_ID", "label", "n_sectors", "sectors"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()