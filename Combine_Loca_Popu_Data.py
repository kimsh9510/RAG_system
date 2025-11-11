"""
Combine_Loca_Popu_Data.py

High-level purpose (beginner-friendly):
- Read a locations CSV (one administrative level) and a set of statistical CSVs
    (or a folder of CSVs). Normalize the stats into a tidy table, pivot stat items
    into columns, and merge them with the location table to produce a single
    combined CSV keyed by administrative area code.

Key behaviors:
- Tries multiple encodings when reading CSV files because sources may use
    different encodings (UTF-8, cp949/EUC-KR, etc.).
- Uses fuzzy column detection for common Korean/English headers such as
    행정구역코드 / area_code, 통계항목 / stat_item, 통계값 / stat_value.
- When possible, calls `Process_Population_Data.main()` to generate a
    combined population file before merging.

The script focuses on readability and helpful errors for newcomers.
"""
from pathlib import Path
import argparse
import sys
from typing import Optional, List, Any
import pandas as pd

def _read_csv_with_encodings(path: Path, **kwargs):
    # Try several likely encodings and return the DataFrame and successful
    # encoding. If none of the encodings work, raise the last exception so
    # callers get a useful error message.
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
            # for other read errors (like parser errors) remember the error
            # and still try the next encoding on the chance it was encoding.
            last_exc = e
            continue
    # If we reach here, none of the encodings worked; raise the last error.
    raise last_exc if last_exc is not None else UnicodeDecodeError("utf-8", b"", 0, 1, "unknown encoding error")


def _find_col(df, candidates: List[str]) -> Optional[str]:
    # Fuzzy-find a column name from a list of candidate strings.
    if df is None:
        return None
    cols = {c.strip().lower(): c for c in df.columns}
    # 1) exact (case-insensitive) match
    for cand in candidates:
        if not cand:
            continue
        lc = cand.strip().lower()
        if lc in cols:
            return cols[lc]
    # 2) substring match
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
    # Accept either a single CSV file or a directory containing many CSVs.
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

    # detect relevant columns using fuzzy matching for common headers
    year_col = _find_col(stats, ["기준년도", "year", "년도"])
    code_col = _find_col(stats, ["행정구역코드", "area_code", "지역코드", "code"])
    item_col = _find_col(stats, ["통계항목", "stat_item", "item", "통계항목명"])
    value_col = _find_col(stats, ["통계값", "stat_value", "value", "값"])

    if code_col is None or item_col is None or value_col is None:
        raise ValueError("Could not find required columns in stats CSVs. Need columns like 행정구역코드, 통계항목, 통계값")

    # Build a normalized output table with clear column names.
    out = pd.DataFrame()
    out['year'] = stats[year_col] if year_col in stats.columns else None
    out['area_code'] = stats[code_col]
    out['stat_item'] = stats[item_col]
    out['stat_value'] = stats[value_col]

    # Trim whitespace to avoid spurious mismatches later.
    out['area_code'] = out['area_code'].astype(str).str.strip()
    out['stat_item'] = out['stat_item'].astype(str).str.strip()
    out['stat_value'] = out['stat_value'].astype(str).str.strip()

    return out


def pivot_stats(stats_df: Any) -> Any:
    # Pivot stat_item into columns; if multiple years or duplicate items exist,
    # pick the first non-empty value. This keeps the pivot simple and predictable.
    stats_df = stats_df.copy()
    pivot = stats_df.pivot_table(
        index='area_code',
        columns='stat_item',
        values='stat_value',
        aggfunc=lambda x: next((v for v in x if str(v).strip() != ''), None)
    )
    pivot = pivot.reset_index()
    # flatten MultiIndex columns if any and ensure all column names are strings
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

    # 3) Run population processing script to produce combined_population.csv
    print('\n--- Running population data processing (Process_Population_Data.py)')
    # location where both final CSVs are expected
    out_dir = Path(__file__).resolve().parents[0] / 'Location_Population_Data'
    try:
        # import and call the main function from Process_Population_Data if available
        try:
            import Process_Population_Data as popproc
        except Exception:
            popproc = None

        if popproc and hasattr(popproc, 'main'):
            stats_dir = Path(__file__).resolve().parents[0] / 'Dataset' / '경계' / '경계' / '통계청_SGIS 행정구역 통계 및 경계_20240630' / '1. 통계' / '1. 2023년 행정구역 통계(인구)'
            out_pop = Path(out_dir) / 'combined_population.csv'
            try:
                popproc.main(stats_dir, out_pop)
            except Exception as e:
                print(f"Population processing script returned an error: {e}")
        else:
            print("Process_Population_Data module not importable or missing 'main'; skipping population processing.")
    except Exception as e:
        print(f"Failed to run population processing: {e}")

    # 4) Merge combined_locations.csv and combined_population.csv on area_code
    print('\n--- Merging combined_locations.csv and combined_population.csv')
    try:
        # ensure pandas is available (module-level import at top should have set pd)
        if pd is None:
            raise RuntimeError('pandas is required to merge CSVs; please install pandas')

        loc_path = Path(out_dir) / 'combined_locations.csv'
        pop_path = Path(out_dir) / 'combined_population.csv'

        if not loc_path.exists():
            raise FileNotFoundError(f"Locations CSV not found: {loc_path}")
        if not pop_path.exists():
            raise FileNotFoundError(f"Population CSV not found: {pop_path}")

        # helper: try multiple encodings when reading
        def _read_with_encodings(p: Path):
            encs = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'latin1']
            last = None
            for e in encs:
                try:
                    return pd.read_csv(p, encoding=e, dtype=str)
                except Exception as ex:
                    last = ex
                    continue
            raise last

        loc_df = _read_with_encodings(loc_path)
        pop_df = _read_with_encodings(pop_path)

        # ensure area_code column exists in both; try common alternatives
        def find_area_col(df):
            for c in df.columns:
                if str(c).strip().lower() in ('area_code', '지역코드', 'adm_cd', 'adm_cd'):
                    return c
            # try contains
            for c in df.columns:
                if 'area' in str(c).lower() or '지역' in str(c) or 'adm' in str(c).lower():
                    return c
            return None

        loc_area = find_area_col(loc_df)
        pop_area = find_area_col(pop_df)
        if loc_area is None or pop_area is None:
            raise ValueError(f"Could not find area_code column in one of the files (found: {loc_area}, {pop_area})")

        # normalize column name
        loc_df['area_code'] = loc_df[loc_area].astype(str).str.strip()
        pop_df['area_code'] = pop_df[pop_area].astype(str).str.strip()

        merged_df = loc_df.merge(pop_df, how='left', on='area_code')
        merged_out = Path(out_dir) / 'combined_locations_population.csv'
        merged_df.to_csv(merged_out, index=False, encoding='utf-8-sig')
        print(f"Wrote merged dataset to: {merged_out}")
    except Exception as e:
        print(f"Error during merge: {e}")

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
    parser.add_argument('--locations', type=str, default=str(Path(__file__).resolve().parents[0] / 'Location_Population_Data' / 'combined_locations.csv'), help='Path to locations CSV (default: Location_Population_Data/combined_locations.csv)')
    parser.add_argument('--stats', type=str, default=str(Path(__file__).resolve().parents[0] / 'Dataset' / '경계' / '경계' / '통계청_SGIS 행정구역 통계 및 경계_20240630' / '1. 통계' / '1. 2023년 행정구역 통계(인구)'), help='Path to stats CSV or directory containing CSVs')
    parser.add_argument('--out', type=str, default=str(Path(__file__).resolve().parents[0] / 'Location_Population_Data' / 'combined_locations_population.csv'), help='Output CSV path')
    args = parser.parse_args()

    try:
        main(Path(args.locations), Path(args.stats), Path(args.out))
    except Exception as e:
        print('Error:', e, file=sys.stderr)
        sys.exit(1)
