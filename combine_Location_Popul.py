"""Combine location metadata with statistical population CSVs.

Reads a location file (Location_Data/combined_locations.csv by default)
and one or more statistical CSVs (directory or single file) that contain
columns similar to: 기준년도, 행정구역코드, 통계항목, 통계값.

The script pivots the statistical data (통계항목 -> columns) and merges
on the administrative code (행정구역코드 == area_code). The result is
written as CSV to the specified output path.

Usage:
  python combine_Location_Popul.py \
    --locations Location_Data/combined_locations.csv \
    --stats Dataset/경계/경계/통계청_SGIS\ 행정구역\ 통계\ 및\ 경계_20240630/1. 통계/1. 2023년 행정구역 통계(인구) \
    --out Location_Data/combined_locations_population.csv

If --stats points to a directory, all .csv files found there will be read
and concatenated (useful because the SGIS folder contains many CSVs).
"""
from pathlib import Path
import argparse
import sys
from typing import Optional, List, Any

try:
    import pandas as pd
except Exception:
    pd = None


def _read_csv_with_encodings(path: Path, **kwargs):
    """Try reading CSV with several encodings and return the dataframe and used encoding.

    kwargs are passed to pd.read_csv.
    Raises the last exception if none of the encodings worked.
    """
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]
    last_exc = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            return df, enc
        except UnicodeDecodeError as e:
            last_exc = e
            # try next encoding
            continue
        except Exception as e:
            # for other read errors (like parser errors) bubble up
            last_exc = e
            # but still try next encoding in case encoding was the issue
            continue
    # If we reach here, none of the encodings worked
    raise last_exc if last_exc is not None else UnicodeDecodeError("utf-8", b"", 0, 1, "unknown encoding error")


def _find_col(df, candidates: List[str]) -> Optional[str]:
    if df is None:
        return None
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if not cand:
            continue
        lc = cand.strip().lower()
        if lc in cols:
            return cols[lc]
    # contains
    for cand in candidates:
        if not cand:
            continue
        lc = cand.strip().lower()
        for k in cols:
            if lc in k:
                return cols[k]
    return None


def read_stats(stats_path: Path) -> Any:
    """Read a single stats CSV or concatenate all CSVs in a directory.

    Returns a DataFrame with normalized columns: ['year','area_code','stat_item','stat_value']
    """
    if stats_path.is_dir():
        files = sorted(stats_path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in stats directory: {stats_path}")
        parts = []
        for f in files:
            try:
                df, enc = _read_csv_with_encodings(f, dtype=str, keep_default_na=False)
            except Exception as e:
                raise RuntimeError(f"Failed to read stats CSV {f}: {e}")
            parts.append(df)
        stats = pd.concat(parts, ignore_index=True, sort=False)
    else:
        try:
            stats, enc = _read_csv_with_encodings(stats_path, dtype=str, keep_default_na=False)
        except Exception as e:
            raise RuntimeError(f"Failed to read stats CSV {stats_path}: {e}")

    # detect relevant columns
    year_col = _find_col(stats, ["기준년도", "year", "년도"])
    code_col = _find_col(stats, ["행정구역코드", "area_code", "지역코드", "code"])
    item_col = _find_col(stats, ["통계항목", "stat_item", "item", "통계항목명"])
    value_col = _find_col(stats, ["통계값", "stat_value", "value", "값"])

    if code_col is None or item_col is None or value_col is None:
        raise ValueError("Could not find required columns in stats CSVs. Need columns like 행정구역코드, 통계항목, 통계값")

    out = pd.DataFrame()
    out['year'] = stats[year_col] if year_col in stats.columns else None
    out['area_code'] = stats[code_col]
    out['stat_item'] = stats[item_col]
    out['stat_value'] = stats[value_col]

    # clean whitespace
    out['area_code'] = out['area_code'].astype(str).str.strip()
    out['stat_item'] = out['stat_item'].astype(str).str.strip()
    out['stat_value'] = out['stat_value'].astype(str).str.strip()

    return out


def pivot_stats(stats_df: Any) -> Any:
    # Pivot stat_item into columns; if multiple years or duplicate items exist, keep first non-empty
    stats_df = stats_df.copy()
    # If there is a year column with multiple years, prefer the latest year per item if available.
    # For now we'll ignore year during pivot; users can filter by year beforehand if needed.

    pivot = stats_df.pivot_table(index='area_code', columns='stat_item', values='stat_value', aggfunc=lambda x: next((v for v in x if str(v).strip() != ''), None))
    pivot = pivot.reset_index()
    # flatten MultiIndex columns if any
    pivot.columns.name = None
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot


def main(locations_csv: Path, stats_path: Path, out_csv: Path):
    if pd is None:
        print("pandas is required. Please install it: pip install pandas", file=sys.stderr)
        sys.exit(2)

    if not locations_csv.exists():
        raise FileNotFoundError(f"Locations file not found: {locations_csv}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats path not found: {stats_path}")

    # Read locations
    try:
        loc, loc_enc = _read_csv_with_encodings(locations_csv, dtype=str, keep_default_na=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read locations CSV {locations_csv}: {e}")

    # canonical area code column in locations
    area_col = None
    for cand in ["area_code", "area_code", "area_code", "area_code", "area_code", "area_code", "area_code", "area_code", 'area_code']:
        if cand in loc.columns:
            area_col = cand
            break
    # try fuzzy match for Korean name
    if area_col is None:
        candidates = ["area_code", "지역코드", "area code", "area.code", "area.code"]
        area_col = _find_col(loc, candidates)

    if area_col is None:
        raise ValueError("Could not determine area code column in locations CSV (expected 'area_code' or '지역코드').")

    loc['area_code'] = loc[area_col].astype(str).str.strip()

    # Read stats and pivot
    stats = read_stats(stats_path)
    stats_wide = pivot_stats(stats)

    # Merge
    merged = loc.merge(stats_wide, how='left', on='area_code')

    # Write output
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"Wrote combined file to: {out_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine location metadata with population/statistics CSVs')
    parser.add_argument('--locations', type=str, default=str(Path(__file__).resolve().parents[0] / 'Location_Data' / 'combined_locations.csv'), help='Path to locations CSV (default: Location_Data/combined_locations.csv)')
    parser.add_argument('--stats', type=str, default=str(Path(__file__).resolve().parents[0] / 'Dataset' / '경계' / '경계' / '통계청_SGIS 행정구역 통계 및 경계_20240630' / '1. 통계' / '1. 2023년 행정구역 통계(인구)'), help='Path to stats CSV or directory containing CSVs')
    parser.add_argument('--out', type=str, default=str(Path(__file__).resolve().parents[0] / 'Location_Data' / 'combined_locations_population.csv'), help='Output CSV path')
    args = parser.parse_args()

    try:
        main(Path(args.locations), Path(args.stats), Path(args.out))
    except Exception as e:
        print('Error:', e, file=sys.stderr)
        sys.exit(1)
