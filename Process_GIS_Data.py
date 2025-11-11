"""
Process_GIS_Data.py

This script reads the merged CSV `Location_Population_Data/combined_locations_population.csv`
and produces two outputs in one pass:

- `Location_Population_Data/combined_locations_population.geojson` — a properties-only
    GeoJSON FeatureCollection (no geometry) where each feature.properties contains
    demographic and population attributes for an administrative area.
- `Location_Population_Data/location_query_result.txt` — a small human-readable
    report filtered by optional location variables parsed from `main.py`.

Behavior notes for beginners:
- The script intentionally avoids reading shapefiles and relies only on the
    combined CSV. It runs `Combine_Loca_Popu_Data.py` first (if present) to
    ensure the combined CSV exists.
- Column detection is heuristic: helpers try common English and Korean column
    names and fall back gracefully when a column is missing.
"""

from pathlib import Path
import pandas as pd
import json
import re
import subprocess
import sys

def _choose_col(df, candidates):
    """Pick the best-matching column from `candidates`.

    1. Try exact (case-sensitive) matches first.
    2. Then try case-insensitive substring matches.
    Returns the actual DataFrame column name or None if nothing matches.
    """
    for c in candidates:
        if c in df.columns:
            return c
    lc = [col.lower() for col in df.columns]
    for c in candidates:
        for i, col in enumerate(lc):
            if c.lower() in col:
                return df.columns[i]
    return None


def _age_group_columns(df):
    # Find columns that represent age groups using a few common patterns. The
    # function returns a list of column names sorted by the trailing number
    # where present (so age_group_001, age_group_002, ... will be ordered).
    pattern1 = re.compile(r'age[_\s-]?group[_\s-]?(\d+)', re.IGNORECASE)
    pattern2 = re.compile(r'^(to_in_|age_group_|agegroup_|age_).*\d+', re.IGNORECASE)
    matches = []
    for c in df.columns:
        if pattern1.search(c) or pattern2.search(c) or ('age' in c.lower() and any(ch.isdigit() for ch in c)):
            matches.append(c)

    # try to sort by trailing numeric suffix when present so age buckets are ordered
    def key(c):
        m = re.search(r'(\d+)(?!.*\d)', c)
        if m:
            return int(m.group(1))
        return c

    return sorted(matches, key=key)


def _read_locations_from_main(root: Path):
    main_path = root / 'main.py'
    if not main_path.exists():
        return None, None, None
    try:
        import ast
        src = main_path.read_text(encoding='utf-8')
        tree = ast.parse(src)
        vals = {}
        # Walk top-level assignments in main.py and attempt to read literal
        # values for location_si/location_gu/location_dong. We avoid importing
        # main.py to prevent executing arbitrary code; instead we parse the AST
        # and use literal_eval when possible.
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ('location_si', 'location_gu', 'location_dong'):
                        try:
                            vals[target.id] = ast.literal_eval(node.value)
                        except Exception:
                            # For non-literal AST nodes try a couple of attributes
                            # that sometimes hold simple values in older ASTs.
                            try:
                                if hasattr(node.value, 's'):
                                    vals[target.id] = node.value.s
                                elif hasattr(node.value, 'value'):
                                    vals[target.id] = node.value.value
                                else:
                                    vals[target.id] = None
                            except Exception:
                                vals[target.id] = None
        return vals.get('location_si'), vals.get('location_gu'), vals.get('location_dong')
    except Exception:
        return None, None, None


def _num(val):
    # Safely parse a numeric-looking value to float. Accepts strings like
    # '1,234' and treats empty/None/NaN as 0.0 without raising.
    try:
        if val is None or val == '' or pd.isna(val):
            return 0.0
        return float(str(val).replace(',', '').strip())
    except Exception:
        return 0.0


def process_csv_and_write_outputs():
    repo_root = Path(__file__).resolve().parents[0]

    # Ensure the combined CSV exists by running the combine script first.
    # Running the combine script here keeps this module self-contained: if the
    # CSV needs to be generated we run the generator, otherwise the call is
    # cheap and quick.
    combine_script = repo_root / 'Combine_Loca_Popu_Data.py'
    if combine_script.exists():
        try:
            subprocess.run([sys.executable, str(combine_script)], cwd=str(repo_root), check=True)
        except subprocess.CalledProcessError as e:
            # Bubble up a clear error so callers know the combine step failed.
            raise RuntimeError(f"Combine_Loca_Popu_Data.py failed: {e}") from e
    else:
        # No combine script found; proceed — the CSV check below will fail
        # with a clear message if the CSV truly is missing.
        pass

    csv_path = repo_root / 'Location_Population_Data' / 'combined_locations_population.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Required CSV not found: {csv_path}")

    # Read the merged CSV that should already contain combined location and
    # population columns. We read everything as strings and convert numbers
    # explicitly to avoid surprises with pandas auto-conversion.
    df = pd.read_csv(csv_path, dtype=str, encoding='utf-8', low_memory=False)

    # detect likely columns in the CSV
    code_col = _choose_col(df, ['area_code', 'area code', '지역코드', 'code'])
    city_col = _choose_col(df, ['city_name', 'sido', 'SIDO_NM', 'city'])
    district_col = _choose_col(df, ['district_name', 'sigungu', 'SIGUNGU_NM', 'district'])
    area_name_col = _choose_col(df, ['area_name', 'area', 'ADM_NM', 'area_name', 'name'])

    total_col = _choose_col(df, ['total_population', '총 인구', '총인구', 'to_in_001', 'in_001'])
    avg_col = _choose_col(df, ['average_age', '평균 나이', 'to_in_002'])
    dens_col = _choose_col(df, ['population_density', '인구밀도', 'to_in_003'])
    aging_col = _choose_col(df, ['aging_index', '노령화지수', 'to_in_004', 'aging_index'])
    elder_dep_col = _choose_col(df, ['elderly_dependency_ratio', 'elderly_dependency'])
    youth_dep_col = _choose_col(df, ['youth_dependency_ratio', 'youth_dependency'])
    male_col = _choose_col(df, ['total_population_male', 'male'])
    female_col = _choose_col(df, ['total_population_female', 'female'])

    age_cols = _age_group_columns(df)

    # read desired location parts from main.py (safe AST parse). These are
    # optional filters used to produce a targeted text report below.
    location_si, location_gu, location_dong = _read_locations_from_main(repo_root)

    # Filter the merged CSV using the optional location variables extracted
    # from main.py. We perform case-insensitive substring matching so users
    # can provide partial names (e.g., 'Seoul' or '서울').
    matched = df
    try:
        if city_col and location_si:
            matched = matched[matched[city_col].astype(str).str.contains(str(location_si), case=False, na=False)]
        if district_col and location_gu:
            matched = matched[matched[district_col].astype(str).str.contains(str(location_gu), case=False, na=False)]
        if area_name_col and location_dong:
            matched = matched[matched[area_name_col].astype(str).str.contains(str(location_dong), case=False, na=False)]
    except Exception:
        matched = df

    # Build a list of property dictionaries (one per matched CSV row). We do
    # explicit numeric parsing so the output types are predictable.
    records = []
    for _, row in matched.iterrows():
        area_code = str(row.get(code_col, '')).strip() if code_col else ''
        city_val = row.get(city_col, '') if city_col else ''
        district_val = row.get(district_col, '') if district_col else ''
        area_val = row.get(area_name_col, '') if area_name_col else ''

        age_wise = [int(_num(row[c])) if c in row.index else 0 for c in age_cols]

        rec = {
            'administrative_level': '동',
            'area_code': area_code,
            'area_name': f"{city_val} / {district_val} / {area_val}",
            'age_wise_population': age_wise,
            'total_population': int(_num(row[total_col])) if total_col and total_col in row.index else 0,
            'average_age': float(_num(row[avg_col])) if avg_col and avg_col in row.index else 0.0,
            'population_density': float(_num(row[dens_col])) if dens_col and dens_col in row.index else 0.0,
            'aging_index': float(_num(row[aging_col])) if aging_col and aging_col in row.index else 0.0,
            'elderly_dependency_ratio': float(_num(row[elder_dep_col])) if elder_dep_col and elder_dep_col in row.index else 0.0,
            'youth_dependency_ratio': float(_num(row[youth_dep_col])) if youth_dep_col and youth_dep_col in row.index else 0.0,
            'total_population_male': int(_num(row[male_col])) if male_col and male_col in row.index else 0,
            'total_population_female': int(_num(row[female_col])) if female_col and female_col in row.index else 0
        }
        records.append(rec)

    # Build GeoJSON FeatureCollection containing only properties (no geometry).
    features = [{
        'type': 'Feature',
        'properties': rec
    } for rec in records]

    feature_collection = {
        'type': 'FeatureCollection',
        'features': features
    }

    out_geojson = repo_root / 'Location_Population_Data' / 'combined_locations_population.geojson'
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    # write GeoJSON
    with open(out_geojson, 'w', encoding='utf-8') as fh:
        json.dump(feature_collection, fh, ensure_ascii=False, indent=2)

    # Write a simple text report filtered by requested location parts (from main.py).
    # This small TXT file is handy for quickly checking the values for the
    # requested location without opening the full CSV/GeoJSON.
    def contains_ignore_case(hay, needle):
        try:
            return needle.lower() in hay.lower()
        except Exception:
            return False

    matched_for_query = []
    for rec in records:
        name = str(rec.get('area_name', ''))
        ok = True
        if location_si and not contains_ignore_case(name, str(location_si)):
            ok = False
        if location_gu and not contains_ignore_case(name, str(location_gu)):
            ok = False
        if location_dong and not contains_ignore_case(name, str(location_dong)):
            ok = False
        if ok:
            matched_for_query.append(rec)

    out_lines = []
    out_lines.append(f"Querying for: {location_si} / {location_gu} / {location_dong}")
    if not matched_for_query:
        out_lines.append("No matching feature found in GeoJSON for requested location")
    else:
        out_lines.append(f"Found {len(matched_for_query)} matching feature(s). Showing first:")
        row = matched_for_query[0]
        out_lines.append(f"administrative_level: {row.get('administrative_level')}")
        out_lines.append(f"area_code: {row.get('area_code')}")
        out_lines.append(f"area_name: {row.get('area_name')}")
        out_lines.append(f"age_wise_population: {row.get('age_wise_population')}")
        out_lines.append(f"total_population: {row.get('total_population')}")
        out_lines.append(f"average_age: {row.get('average_age')}")
        out_lines.append(f"population_density: {row.get('population_density')}")
        out_lines.append(f"aging_index: {row.get('aging_index')}")
        out_lines.append(f"elderly_dependency_ratio: {row.get('elderly_dependency_ratio')}")
        out_lines.append(f"youth_dependency_ratio: {row.get('youth_dependency_ratio')}")
        out_lines.append(f"total_population_male: {row.get('total_population_male')}")
        out_lines.append(f"total_population_female: {row.get('total_population_female')}")

    out_path = repo_root / 'Location_Population_Data' / 'location_query_result.txt'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out_lines))

    # summary statistics print
    print("\n=== Summary Statistics ===")
    total_pop_sum = sum(int(rec.get('total_population', 0) or 0) for rec in records)
    print(f"Features produced: {len(records)}")
    print(f"Total population: {total_pop_sum:,.0f}")

    print(f"Wrote GeoJSON to: {out_geojson}")
    print(f"Wrote query result to: {out_path}")

    return records


if __name__ == '__main__':
    print('Processing CSV and creating GeoJSON + TXT (single pass)')
    process_csv_and_write_outputs()
    print('\n✓ Processing complete!')
