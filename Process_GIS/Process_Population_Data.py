"""
Combine SGIS population CSVs into one merged dataset keyed by 행정구역코드 (area code).

This script is intended to be readable and beginner-friendly. It performs these high-
level steps:

- Read all CSV files from a statistics directory (tries several common encodings).
- Detect the column names that contain: year, area code, statistic item, and value.
- Normalize each CSV into a small table with columns: year, area_code, code, value.
    - The `code` column contains a short SGIS variable code when it can be extracted
        (for example, `to_in_001` or `in_age_001`). If a code cannot be parsed, the
        raw item text is used as a fallback.
- Concatenate all CSVs into one table and pivot it so each variable/code becomes
    its own column, with one row per `area_code`.
- Rename known SGIS variable codes to readable column names (e.g. `to_in_001`
    -> `total_population`) and canonicalize others.

Outputs a CSV (default: Location_Population_Data/combined_population.csv).

Notes for beginners:
- The code tries multiple encodings because Korean CSVs often come in different
    encodings (UTF-8, cp949, euc-kr, etc.).
- Column name detection is fuzzy: it looks for common Korean and English names,
    and also does a lowercase substring match as a fallback.
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
    # Try reading the CSV using several common encodings used for Korean data.
    # Return the DataFrame and the encoding that worked.
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]
    last_exc = None
    for enc in encodings:
        try:
            # keep_default_na=False avoids turning empty strings into NaN automatically.
            df = pd.read_csv(path, encoding=enc, **kwargs)
            return df, enc
        except Exception as e:
            last_exc = e
            # Try the next encoding
            continue
    # If none of the encodings worked, raise the last encountered exception.
    raise last_exc


def _find_col(df, candidates: List[str]) -> Optional[str]:
    """Find the best matching column name from `candidates`.

    - First tries exact (case-insensitive) matches.
    - Then falls back to substring matches (candidate is contained in an
      existing column name).

    Returns the actual column name from the DataFrame or None.
    """
    if df is None:
        return None
    # Map normalized column name -> original column name
    cols = {c.strip().lower(): c for c in df.columns}

    # 1) Exact match on normalized candidate
    for cand in candidates:
        if not cand:
            continue
        lc = cand.strip().lower()
        if lc in cols:
            return cols[lc]

    # 2) Substring match: sometimes headers contain extra text
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
    # Try to find common SGIS-style codes inside the string. We try a strict
    # 3-digit match first, then a more permissive lowercase match.
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
    # Verify directory and CSV files
    if not stats_dir.exists():
        raise FileNotFoundError(f"Stats directory not found: {stats_dir}")
    files = sorted(stats_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {stats_dir}")

    parts = []
    for f in files:
        # Read each CSV robustly trying typical encodings
        try:
            df, enc = _read_csv_with_encodings(f, dtype=str, keep_default_na=False)
        except Exception as e:
            raise RuntimeError(f"Failed to read {f}: {e}")

        # Detect useful columns using fuzzy matching (Korean + English variants)
        year_col = _find_col(df, ["기준년도", "year", "년도"])
        code_col = _find_col(df, ["행정구역코드", "area_code", "지역코드", "ADM_CODE"])
        item_col = _find_col(df, ["통계항목", "stat_item", "item", "통계항목명", "변수코드", "표준변수코드", "변수"])
        value_col = _find_col(df, ["통계값", "stat_value", "value", "값"])

        # If required fields cannot be found, skip this file but warn the user.
        if code_col is None or item_col is None or value_col is None:
            print(f"Warning: skipping {f} because required columns not found (tried code/item/value).", file=sys.stderr)
            continue

        # Standardize the CSV into a small DataFrame with known column names.
        sub = pd.DataFrame()
        sub['year'] = df[year_col] if year_col in df.columns else None
        sub['area_code'] = df[code_col].astype(str).str.strip()

        # Keep raw item text and try to extract a short SGIS variable code
        sub['raw_item'] = df[item_col].astype(str)
        sub['code'] = sub['raw_item'].apply(lambda x: extract_code_from_text(x) or x.strip())
        sub['value'] = df[value_col].astype(str).str.strip()

        parts.append(sub[['year', 'area_code', 'code', 'value']])

    if not parts:
        raise RuntimeError(f"No usable population CSVs found in {stats_dir}")

    # Concatenate all standardized parts into a single long-form DataFrame.
    all_stats = pd.concat(parts, ignore_index=True, sort=False)
    return all_stats


def pivot_and_rename(all_stats: Any) -> Any:
    """Pivot the long-form stats table into a wide table and rename columns.

    Input: all_stats with columns ['year','area_code','code','value']
    Output: DataFrame indexed by 'area_code' with one column per 'code'.
    """
    # Pivot so each 'code' becomes a separate column. The aggfunc picks the
    # first non-empty value if duplicates exist for the same area_code/code.
    pivot = all_stats.pivot_table(
        index='area_code',
        columns='code',
        values='value',
        aggfunc=lambda x: next((v for v in x if str(v).strip() != ''), None)
    )
    pivot = pivot.reset_index()

    # Rename known SGIS codes to readable names and canonicalize others.
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
    # Read source CSVs and normalize them into a long-form DataFrame.
    # This step may take a little time depending on how many CSVs are present.
    all_stats = read_and_standardize_stats(stats_dir)

    # Pivot the long-form table into a wide table where each SGIS code becomes
    # a separate column. The result has one row per administrative area code.
    pivot = pivot_and_rename(all_stats)

    # Ensure the output directory exists and write the result using
    # utf-8-sig encoding (includes a BOM) which helps Excel detect UTF-8
    # correctly on Windows systems frequently used in Korean locales.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"Wrote combined population CSV to: {out_csv}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Combine SGIS population CSVs into a single file keyed by 행정구역코드')
    process_popu_root = Path(__file__).resolve().parents[0]     # Process_GIS 폴더
    project_root = process_popu_root.parents[0]
         
    default_stats = project_root / 'Dataset' / '경계' / '경계' / '통계청_SGIS 행정구역 통계 및 경계_20240630' / '1. 통계' / '1. 2023년 행정구역 통계(인구)'
    default_out = project_root / 'Dataset' / 'GIS_data' / 'combined_population.csv'
    p.add_argument('--stats-dir', type=str, default=str(default_stats), help='Directory containing population CSVs')
    p.add_argument('--out', type=str, default=str(default_out), help='Output CSV path')
    args = p.parse_args()

    try:
        main(Path(args.stats_dir), Path(args.out))
    except Exception as e:
        print('Error:', e, file=sys.stderr)
        sys.exit(1)
