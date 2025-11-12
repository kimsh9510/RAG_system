"""
Process_GIS_Data.py  — DONG-ONLY OUTPUT

What this script does
- Runs Combine_Loca_Popu_Data.main (if needed) to ensure
  Location_Population_Data/combined_locations_population.csv exists.
- Extracts location variables (location_si, location_gu, location_dong) from main.py.
- Reads the combined CSV and filters EXACTLY ONE administrative level: 동 (dong) ONLY.
- Writes ONLY the matched dong into:
    - Location_Population_Data/location_query_result.txt
    - Location_Population_Data/location_query_result.geojson
      (FeatureCollection with a single Feature; geometry is null)

Notes
- If summary columns aren’t confidently found, we fallback to the last 8 columns of the CSV
  and map them in this order:
    total_population, average_age, population_density, aging_index,
    elderly_dependency_ratio, youth_dependency_ratio,
    total_population_male, total_population_female
- GeoJSON features use null geometry (attach real geometry later if available).
"""
from pathlib import Path
import ast
import json
import sys
from typing import Dict, List

# Local imports guarded below to allow useful error messages
try:
    import pandas as pd
except Exception:
    pd = None


def _extract_locations_from_main(main_path: Path) -> Dict[str, str]:
    """Parse `main.py` and return location_si, location_gu, location_dong if present."""
    text = main_path.read_text(encoding='utf-8')
    tree = ast.parse(text)
    found = {}
    targets = {"location_si", "location_gu", "location_dong"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in targets:
                    value = node.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        found[target.id] = value.value
                    elif isinstance(value, ast.Str):
                        found[target.id] = value.s
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in targets:
            val = node.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                found[node.target.id] = val.value
            elif isinstance(val, ast.Str):
                found[node.target.id] = val.s

    # Fallback: try importing (many projects define globals at import-time)
    if len(found) < 3:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location('proj_main', str(main_path))
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                for name in list(targets - set(found.keys())):
                    if hasattr(mod, name):
                        v = getattr(mod, name)
                        if isinstance(v, str):
                            found[name] = v
            except Exception:
                pass
        except Exception:
            pass

    return {k: found.get(k) for k in ("location_si", "location_gu", "location_dong")}


def _read_csv_with_encodings(path: Path):
    encs = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'latin1']
    last = None
    for e in encs:
        try:
            return pd.read_csv(path, encoding=e, dtype=str)
        except Exception as ex:
            last = ex
    raise last


def _find_column_by_candidates(cols: List[str], candidates: List[str]):
    for cand in candidates:
        for k in cols:
            if cand.lower() == k.lower():
                return k
    # substring match
    for cand in candidates:
        lc = cand.lower()
        for k in cols:
            if lc in k.lower():
                return k
    return None


def _map_summary_columns(cols: List[str]) -> Dict[str, str]:
    """Map requested summary fields to actual CSV column names (with fallback)."""
    wants = [
        ('total_population', ['total_population', 'tot_pop', '총인구', 'total']),
        ('average_age', ['average_age', 'avg_age', '평균나이', 'average age', 'mean_age']),
        ('population_density', ['population_density', '인구밀도', 'density']),
        ('aging_index', ['aging_index', '노령화지수', 'aging index']),
        ('elderly_dependency_ratio', ['elderly_dependency_ratio', '노년부양비', 'elderly_dependency']),
        ('youth_dependency_ratio', ['youth_dependency_ratio', '유년부양비', 'youth_dependency']),
        ('total_population_male', ['total_population_male', 'male', '남자', 'males']),
        ('total_population_female', ['total_population_female', 'female', '여자', 'females']),
    ]

    mapping = {}
    for key, cands in wants:
        col = _find_column_by_candidates(cols, cands)
        if col:
            mapping[key] = col

    # Fallback: tail 8 columns map in fixed order
    if len(mapping) < len(wants) and len(cols) >= 8:
        tail = cols[-8:]
        fallback_keys = [w[0] for w in wants]
        for k, col in zip(fallback_keys, tail):
            if k not in mapping:
                mapping[k] = col

    return mapping


def _to_numeric(df, cols: List[str]):
    for c in cols:
        if c in df.columns:
            # remove thousands separators, coerce errors
            df[c + '_num'] = pd.to_numeric(df[c].str.replace(',', '').replace('', pd.NA), errors='coerce')
    return df


# column name holders discovered at runtime
cols_hint = {
    'city': None,
    'district': None,
    'area': None,
}


def _extract_dong_row(df: pd.DataFrame, si: str, gu: str, dong: str) -> pd.DataFrame: # pyright: ignore[reportInvalidTypeForm]
    sel = (
        (df[cols_hint['city']] == si) &
        (df[cols_hint['district']] == gu) &
        (df[cols_hint['area']] == dong)
    )
    return df[sel]


def _make_dong_result(sub: pd.DataFrame, cols_map: Dict[str, str]) -> Dict[str, object]: # pyright: ignore[reportInvalidTypeForm]
    """Build a single result dict for the one matched dong (no city/gu aggregates)."""
    if sub.empty:
        return None
    first = sub.iloc[0]
    
    city_name = first.get('city_name') if 'city_name' in sub.columns else None
    district_name = first.get('district_name') if 'district_name' in sub.columns else None
    area_name = first.get('area_name') if 'area_name' in sub.columns else None

    location_name = " ".join([v for v in [city_name, district_name, area_name] if v])

    # try to pass through codes/names if present
    result = {
        'administrative_level': '시도, 시군구, 동',
        'area_code': first.get('area_code') if 'area_code' in sub.columns else None,
        'full_location_name': location_name,
        'total_population': None,
        'average_age': None,
        'population_density': None,
        'aging_index': None,
        'elderly_dependency_ratio': None,
        'youth_dependency_ratio': None,
        'total_population_male': None,
        'total_population_female': None,
    }

    # simple passthrough for totals/single-row stats (already numeric columns exist with _num)
    def sum_or_first_num(key: str, as_int: bool = True):
        coln = cols_map.get(key)
        if not coln:
            return None
        num = coln + '_num'
        if num not in sub.columns:
            return None
        s = sub[num].sum(min_count=1)  # usually a single-row subset, but sum is safe
        if pd.isna(s):
            return None
        if as_int:
            try:
                return int(s)
            except Exception:
                return float(s)
        return float(s)

    # totals
    result['total_population'] = sum_or_first_num('total_population', as_int=True)
    result['total_population_male'] = sum_or_first_num('total_population_male', as_int=True)
    result['total_population_female'] = sum_or_first_num('total_population_female', as_int=True)

    # averages/ratios — for a single dong, mean == value
    for key in ('average_age', 'population_density', 'aging_index',
                'elderly_dependency_ratio', 'youth_dependency_ratio'):
        val = sum_or_first_num(key, as_int=False)
        result[key] = val

    return result


def main():
    root = Path(__file__).resolve().parents[0]
    combine_module = root / 'Combine_Loca_Popu_Data.py'
    main_py = root / 'main.py'
    out_dir = root / 'Location_Population_Data'
    out_dir.mkdir(exist_ok=True)
    combined_csv = out_dir / 'combined_locations_population.csv'

    # 1) ensure pandas
    if pd is None:
        print('pandas is required. Please install: pip install pandas')
        sys.exit(1)

    # 2) Ensure combined CSV exists (run combine if needed)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('combine_mod', str(combine_module))
        combine_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(combine_mod)
        if not combined_csv.exists():
            loc_path = out_dir / 'combined_locations.csv'
            stats_path = root / 'Dataset' / '경계' / '경계' / '통계청_SGIS 행정구역 통계 및 경계_20240630' / '1. 통계' / '1. 2023년 행정구역 통계(인구)'
            try:
                combine_mod.main(loc_path, stats_path, combined_csv)
            except Exception as e:
                print(f'Warning: running combine script raised an error: {e} (continuing if output exists)')
    except Exception as e:
        print(f'Could not import or run Combine_Loca_Popu_Data.py: {e}\nProceeding assuming existing combined CSV.')

    if not combined_csv.exists():
        print(f'combined CSV not found at expected path: {combined_csv}')
        sys.exit(1)

    # 3) Extract locations from main.py
    locs = _extract_locations_from_main(main_py)
    location_si = locs.get('location_si')
    location_gu = locs.get('location_gu')
    location_dong = locs.get('location_dong')

    if not (location_si and location_gu and location_dong):
        print('Could not find all location variables in main.py. Please ensure location_si, location_gu, location_dong are set.')
        print('Found:', locs)
        sys.exit(1)

    # 4) Read CSV
    df = _read_csv_with_encodings(combined_csv)

    # 5) Determine city/district/area columns (robust name finding)
    cols_hint['city'] = _find_column_by_candidates(list(df.columns), ['city_name', '시도', 'city']) or \
                        _find_column_by_candidates(list(df.columns), ['city_name', 'city']) or 'city_name'
    cols_hint['district'] = _find_column_by_candidates(list(df.columns), ['district_name', '시군구', 'district']) or 'district_name'
    cols_hint['area'] = _find_column_by_candidates(list(df.columns), ['area_name', '동', 'area']) or 'area_name'

    # 6) Map summary columns and create numeric clones
    cols_map = _map_summary_columns(list(df.columns))
    df = _to_numeric(df, list(cols_map.values()))

    # 7) Filter ONLY the requested dong
    sub = _extract_dong_row(df, location_si, location_gu, location_dong)
    if sub.empty:
        print('No matching row found for the provided dong (시도, 시군구, 동).')
        print(f'Queried: si="{location_si}", gu="{location_gu}", dong="{location_dong}"')
        sys.exit(1)

    # 8) Build dong-only result
    dong_result = _make_dong_result(sub, cols_map)
    if dong_result is None:
        print('Failed to build dong result (missing columns?).')
        sys.exit(1)

    # 9) Write outputs: TXT (JSON) and GeoJSON (ONE feature only)
    txt_out = out_dir / 'location_query_result.txt'
    geojson_out = out_dir / 'location_query_result.geojson'

    # Keep format as a single-item list to avoid downstream breakage
    with txt_out.open('w', encoding='utf-8') as fh:
        json.dump([dong_result], fh, ensure_ascii=False, indent=2)

    feature = {
        'type': 'Feature',
        'geometry': None,
        'properties': dong_result
    }
    geo = {
        'type': 'FeatureCollection',
        'features': [feature]
    }
    with geojson_out.open('w', encoding='utf-8') as fh:
        json.dump(geo, fh, ensure_ascii=False, indent=2)

    print('Created the files:', txt_out, geojson_out)


if __name__ == '__main__':
    main()
