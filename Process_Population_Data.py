#!/usr/bin/env python3
"""Combine SGIS population CSVs into one merged dataset keyed by 행정구역코드 (area code).

This script will:
- Read all CSV files in the specified SGIS stats directory (default: the 2023 population folder).
- Detect key columns (년도/기준년도, 행정구역코드, 통계항목/변수코드, 통계값).
- Extract or normalize SGIS variable codes (e.g., to_in_001, in_age_001) from fields or filenames.
- Pivot so each SGIS variable code becomes a column and merge all into a single dataframe keyed by 행정구역코드.
- Rename known SGIS codes to meaningful English column names per the mapping provided.

Outputs a CSV (default: Location_Data/population_combined.csv).

Usage:
  python population_Data_Process.py
  python population_Data_Process.py --stats-dir "Dataset/경계/경계/통계청_SGIS 행정구역 통계 및 경계_20240630/1. 통계/1. 2023년 행정구역 통계(인구)" --out Location_Data/population_combined.csv
"""
from pathlib import Path
import argparse
import sys
import re
from typing import List, Optional, Any

try:
    import pandas as pd
except Exception:
    pd = None


def _read_csv_with_encodings(path: Path, **kwargs):
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]
    last_exc = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            return df, enc
        except Exception as e:
            last_exc = e
            continue
    raise last_exc


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
    for cand in candidates:
        if not cand:
            continue
        lc = cand.strip().lower()
        for k in cols:
            if lc in k:
                return cols[k]
    return None


SGIS_CODE_TO_NAME = {
    # mapping provided by user
    "to_in_005": "elderly_dependency_ratio",
    "to_in_004": "aging_index",
    "to_in_006": "youth_dependency_ratio",
    "to_in_003": "population_density",
    "to_in_001": "total_population",
    "to_in_007": "total_population_male",
    "to_in_008": "total_population_female",
    "to_in_002": "average_age",
}


def extract_code_from_text(s: str) -> Optional[str]:
    """Try to extract SGIS variable code from a string, e.g. 'to_in_001' or 'in_age_001'."""
    if not isinstance(s, str):
        return None
    # common patterns: to_in_001, in_age_001, to_in_00X
    m = re.search(r"(to_in_[0-9]{3}|in_age_[0-9]{3}|in_age_[0-9]{1,3}|to_in_[0-9]{1,3})", s)
    if m:
        return m.group(0)
    # sometimes codes appear like TO_IN_001 or IN_AGE_001
    m = re.search(r"(to_in_[0-9]{1,3}|in_age_[0-9]{1,3})", s.lower())
    if m:
        return m.group(0)
    return None


def read_and_standardize_stats(stats_dir: Path) -> Any:
    """Read all CSVs in stats_dir and return a standardized DataFrame with columns:
    ['year','area_code','code','value'] where 'code' is the SGIS variable code when available.
    """
    if not stats_dir.exists():
        raise FileNotFoundError(f"Stats directory not found: {stats_dir}")
    files = sorted(stats_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {stats_dir}")

    parts = []
    for f in files:
        try:
            df, enc = _read_csv_with_encodings(f, dtype=str, keep_default_na=False)
        except Exception as e:
            raise RuntimeError(f"Failed to read {f}: {e}")

        # detect columns
        year_col = _find_col(df, ["기준년도", "year", "년도"])
        code_col = _find_col(df, ["행정구역코드", "area_code", "지역코드", "ADM_CODE"])
        item_col = _find_col(df, ["통계항목", "stat_item", "item", "통계항목명", "변수코드", "표준변수코드", "변수"])
        value_col = _find_col(df, ["통계값", "stat_value", "value", "값"])

        if code_col is None or item_col is None or value_col is None:
            # Try to continue: some CSVs might have different headers; attempt to infer
            # If cannot locate necessary columns, skip with warning
            print(f"Warning: skipping {f} because required columns not found (tried code/item/value).", file=sys.stderr)
            continue

        sub = pd.DataFrame()
        sub['year'] = df[year_col] if year_col in df.columns else None
        sub['area_code'] = df[code_col].astype(str).str.strip()
        # Try to extract a clean SGIS variable code from item column, else use the raw item text
        sub['raw_item'] = df[item_col].astype(str)
        sub['code'] = sub['raw_item'].apply(lambda x: extract_code_from_text(x) or x.strip())
        sub['value'] = df[value_col].astype(str).str.strip()
        parts.append(sub[['year', 'area_code', 'code', 'value']])

    if not parts:
        raise RuntimeError(f"No usable population CSVs found in {stats_dir}")

    all_stats = pd.concat(parts, ignore_index=True, sort=False)
    return all_stats


def pivot_and_rename(all_stats: Any) -> Any:
    # pivot so each 'code' becomes a column
    pivot = all_stats.pivot_table(index='area_code', columns='code', values='value', aggfunc=lambda x: next((v for v in x if str(v).strip() != ''), None))
    pivot = pivot.reset_index()

    # rename known SGIS codes
    rename_map = {}
    for c in list(pivot.columns):
        if c == 'area_code':
            continue
        if c in SGIS_CODE_TO_NAME:
            rename_map[c] = SGIS_CODE_TO_NAME[c]
        else:
            # handle in_age_* series -> make a readable name
            m = re.match(r'in_age_0*([0-9]+)', str(c))
            if m:
                num = m.group(1)
                # produce age group column name like age_group_001
                rename_map[c] = f'age_group_{int(num):03d}'
            else:
                # fallback: canonicalize to a safe column name
                safe = re.sub(r'[^0-9a-zA-Z_]+', '_', str(c)).lower()
                rename_map[c] = safe

    pivot = pivot.rename(columns=rename_map)
    return pivot


def main(stats_dir: Path, out_csv: Path):
    if pd is None:
        print("pandas is required. Install with: pip install pandas", file=sys.stderr)
        sys.exit(2)

    all_stats = read_and_standardize_stats(stats_dir)
    pivot = pivot_and_rename(all_stats)

    # write output
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"Wrote combined population CSV to: {out_csv}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Combine SGIS population CSVs into a single file keyed by 행정구역코드')
    default_stats = Path(__file__).resolve().parents[0] / 'Dataset' / '경계' / '경계' / '통계청_SGIS 행정구역 통계 및 경계_20240630' / '1. 통계' / '1. 2023년 행정구역 통계(인구)'
    default_out = Path(__file__).resolve().parents[0] / 'Location_Population_Data' / 'combined_population.csv'
    p.add_argument('--stats-dir', type=str, default=str(default_stats), help='Directory containing population CSVs')
    p.add_argument('--out', type=str, default=str(default_out), help='Output CSV path')
    args = p.parse_args()

    try:
        main(Path(args.stats_dir), Path(args.out))
    except Exception as e:
        print('Error:', e, file=sys.stderr)
        sys.exit(1)
