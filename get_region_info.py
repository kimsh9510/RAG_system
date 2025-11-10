#!/usr/bin/env python3
"""Utility to extract region information from the provided CSV and print a formatted summary.

Usage:
  - import get_region_info.get_region_summary(...) from other code
  - run as script: python get_region_info.py [sido] [sigungu] [dong] [csv_path]

This script is robust to several column-name variants (Korean/English) and attempts to
parse numeric values safely.
"""
from pathlib import Path
import sys
from typing import Optional, Tuple

try:
    import pandas as pd
except Exception:
    pd = None


_DEFAULT_CSV = Path(__file__).resolve().parents[0] / "Dataset" / "기본데이터" / "지역별_인구_경계_데이터.csv"


def _find_col(df, candidates):
    """Return first matching column name in df for any of the candidate keys (case-insensitive)."""
    if df is None:
        return None
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand is None:
            continue
        lc = cand.lower()
        # exact lower match
        if lc in cols:
            return cols[lc]
    # try contains
    for cand in candidates:
        lc = cand.lower()
        for c in cols:
            if lc in c:
                return cols[c]
    return None


def _to_number(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s2 = str(s).strip()
    if s2 == "":
        return None
    # remove common thousands separators and textual markers
    s2 = s2.replace(",", "").replace(" ", "")
    # handle percent or other
    try:
        return float(s2)
    except Exception:
        # try to extract digits
        import re
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", s2)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
        return None


def _parse_centroid(val: str) -> Optional[Tuple[float, float]]:
    # Try patterns like "lat,lon" or "(lat lon)" or "POINT(lon lat)"
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    s = s.replace("(", "").replace(")", "")
    s = s.replace("POINT", "").replace("point", "")
    # try comma or whitespace
    for sep in [",", " ", "\t"]:
        parts = [p.strip() for p in s.split(sep) if p.strip()]
        if len(parts) >= 2:
            lat = _to_number(parts[0])
            lon = _to_number(parts[1])
            if lat is not None and lon is not None:
                return (lat, lon)
    return None


def get_region_summary(location_si: str, location_gu: str, location_dong: str, csv_path: Optional[Path] = None) -> Optional[str]:
    """Load CSV and return the formatted summary text for the matching region.

    Returns None when no matching row is found.
    """
    path = Path(csv_path) if csv_path else _DEFAULT_CSV
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    if pd is None:
        raise RuntimeError("pandas is required for this utility. Please install pandas.")

    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    # normalize column names by stripping
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # identify columns for matching location fields (try many variants)
    sido_col = _find_col(df, ["시도", "시", "sido", "location_si", "province", "도/시"])
    gu_col = _find_col(df, ["시군구", "구", "sigungu", "gu", "district", "location_gu"])
    dong_col = _find_col(df, ["동", "읍면", "dong", "dong_name", "location_dong", "법정동"])

    # If none of these exist, try to find a single 'area' column that contains all names
    if not any([sido_col, gu_col, dong_col]):
        # attempt to find column likely called 'region' or 'area'
        area_like = _find_col(df, ["지역명", "area_name", "region"])
        if area_like:
            # we'll do substring match against provided three-part name
            mask = df[area_like].astype(str).str.contains(f"{location_si}") & df[area_like].astype(str).str.contains(f"{location_gu}") & df[area_like].astype(str).str.contains(f"{location_dong}")
        else:
            mask = pd.Series([False] * len(df))
    else:
        mask = pd.Series([True] * len(df))
        if sido_col:
            mask = mask & (df[sido_col].astype(str).str.strip() == str(location_si).strip())
        if gu_col:
            mask = mask & (df[gu_col].astype(str).str.strip() == str(location_gu).strip())
        if dong_col:
            mask = mask & (df[dong_col].astype(str).str.strip() == str(location_dong).strip())

    matched = df.loc[mask]
    if matched.empty:
        return None

    # take first match
    row = matched.iloc[0]

    # find columns for requested outputs
    col_map = {
        "area_name": ["지역명", "area_name", "name", "지역_명"],
        "administrative_level": ["행정구역 수준", "administrative_level", "level"],
        "area_code": ["지역코드", "area_code", "code"],
        "population": ["총 인구", "총인구", "population", "pop", "total_population"],
        "density": ["인구 밀도", "인구밀도", "density", "pop_density"],
        "area": ["면적", "면적_km2", "area_km2", "area", "area_sqkm", "area(㎢)", "area_km²"],
        "centroid": ["centroid", "중심 좌표", "centroid_xy", "centroid_latlon", "좌표"],
        "lat": ["위도", "lat", "latitude", "centroid_lat"],
        "lon": ["경도", "lon", "longitude", "centroid_lon"],
        "average_age": ["평균 연령", "average_age", "mean_age", "avg_age"],
        "elderly_index": ["노령화지수", "elderly_index", "elderly_index(%)", "old_age_index"],
    }

    found = {}
    for key, candidates in col_map.items():
        c = _find_col(df, candidates)
        found[key] = c

    # extract values and parse numbers
    area_name = str(row.get(found.get("area_name"), "")).strip()
    administrative_level = str(row.get(found.get("administrative_level"), "")).strip()
    area_code = str(row.get(found.get("area_code"), "")).strip()

    population_raw = row.get(found.get("population")) if found.get("population") else None
    population = _to_number(population_raw) or 0

    density_raw = row.get(found.get("density")) if found.get("density") else None
    density = _to_number(density_raw)

    area_raw = row.get(found.get("area")) if found.get("area") else None
    area_val = _to_number(area_raw)
    area_sqkm = None
    if area_val is not None:
        # heuristic: if area seems large (> 1000) it's likely in m2 -> convert to km2
        if area_val > 1000:
            area_sqkm = area_val / 1_000_000.0
        else:
            # assume already in km2
            area_sqkm = float(area_val)

    # centroid handling: prefer explicit lat/lon cols
    lat = _to_number(row.get(found.get("lat"))) if found.get("lat") else None
    lon = _to_number(row.get(found.get("lon"))) if found.get("lon") else None
    if lat is None or lon is None:
        # try combined centroid column
        cen_col = found.get("centroid")
        if cen_col and str(row.get(cen_col)).strip():
            parsed = _parse_centroid(str(row.get(cen_col)))
            if parsed:
                lat, lon = parsed

    # if density missing but population and area available, compute density
    if density is None and population and area_sqkm and area_sqkm > 0:
        density = population / area_sqkm

    average_age = _to_number(row.get(found.get("average_age"))) or 0.0
    elderly_index = _to_number(row.get(found.get("elderly_index"))) or 0.0

    # prepare formatted output
    try:
        pop_int = int(round(population))
    except Exception:
        pop_int = 0

    density_f = float(density) if (density is not None) else 0.0
    area_f = float(area_sqkm) if (area_sqkm is not None) else 0.0
    centroid_lat = float(lat) if lat is not None else 0.0
    centroid_lon = float(lon) if lon is not None else 0.0

    out = (
        f"지역명: {area_name}\n"
        f"행정구역 수준: {administrative_level}\n"
        f"지역코드: {area_code}\n"
        f"총 인구: {pop_int:,}명\n"
        f"인구 밀도: {density_f:.1f}명/km²\n"
        f"면적: {area_f:.2f}km²\n"
        f"중심 좌표: (위도 {centroid_lat:.4f}, 경도 {centroid_lon:.4f})\n"
        f"평균 연령: {average_age:.1f}세\n"
        f"노령화지수: {elderly_index:.1f}\n"
    )

    return out


if __name__ == "__main__":
    # allow command-line overrides; otherwise use example defaults
    import argparse

    parser = argparse.ArgumentParser(description="Get region info summary from CSV")
    parser.add_argument("sido", nargs="?", default="서울시", help="시도 (e.g., 서울시)")
    parser.add_argument("sigungu", nargs="?", default="서초구", help="시군구 (e.g., 서초구)")
    parser.add_argument("dong", nargs="?", default="서초동", help="동 (e.g., 서초동)")
    parser.add_argument("--csv", dest="csv", default=str(_DEFAULT_CSV), help="path to CSV file")
    args = parser.parse_args()

    try:
        summary = get_region_summary(args.sido, args.sigungu, args.dong, args.csv)
        if summary is None:
            print(f"No matching row found for {args.sido} / {args.sigungu} / {args.dong} in {args.csv}")
            sys.exit(2)
        print(summary)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)
