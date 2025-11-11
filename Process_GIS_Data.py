from pathlib import Path
import pandas as pd
import geopandas as gpd

def _choose_col(df, candidates):
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
    # Find columns that represent age groups. Accept several naming patterns.
    import re
    pattern1 = re.compile(r'age[_\s-]?group[_\s-]?(\d+)', re.IGNORECASE)
    pattern2 = re.compile(r'^(to_in_|age_group_|agegroup_|age_).*\d+', re.IGNORECASE)
    matches = []
    for c in df.columns:
        if pattern1.search(c) or pattern2.search(c) or ('age' in c.lower() and any(ch.isdigit() for ch in c)):
            matches.append(c)

    # try to sort by trailing number when present
    def key(c):
        m = re.search(r'(\d+)(?!.*\d)', c)
        if m:
            return int(m.group(1))
        return c

    return sorted(matches, key=key)


def _choose_gdf_col(gdf, candidates):
    # prefer exact match
    for c in candidates:
        if c in gdf.columns:
            return c
    # case-insensitive contains
    lc = [col.lower() for col in gdf.columns]
    for c in candidates:
        cl = c.lower()
        for i, col in enumerate(lc):
            if cl in col:
                return gdf.columns[i]
    # fallback: try to find any column that looks like a code or name
    for col in gdf.columns:
        if 'cd' in col.lower() or 'code' in col.lower() or 'adm' in col.lower():
            return col
    for col in gdf.columns:
        if 'nm' in col.lower() or 'name' in col.lower() or 'kor' in col.lower():
            return col
    # final fallback: return first column
    return gdf.columns[0] if len(gdf.columns) > 0 else None


def generate_geojson_from_csv_and_shapefiles():

    repo_root = Path(__file__).resolve().parents[0]

    # Do NOT run other scripts here. Expect the merged CSV to already exist.
    # If it's missing, raise a clear error so the caller/user can run the generator separately.

    csv_path = repo_root / 'Location_Population_Data' / 'combined_locations_population.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Required CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, encoding='utf-8', low_memory=False)

    # load only the 행정동 shapefile and determine its code/name columns
    base = repo_root / 'Dataset' / '경계' / '경계' / '통계청_SGIS 행정구역 통계 및 경계_20240630' / '2. 경계'
    dong_shp = base / '3. 2024년 2분기 기준 행정동 경계' / 'bnd_dong_00_2024_2Q.shp'

    dong_gdf = gpd.read_file(str(dong_shp))

    # determine dong code/name columns robustly and create uniform 'code' and 'name' columns
    dong_code_col = _choose_gdf_col(dong_gdf, ['ADM_DR_CD', 'EMD_CD', 'ADM_CD', 'ADMD_CD', 'BJDONG_CD', 'CTP_CD'])
    dong_name_col = _choose_gdf_col(dong_gdf, ['ADM_DR_NM', 'EMD_NM', 'ADM_NM', 'BJDONG_NM', 'NAME'])

    def _norm_gdf_col_to_str(gdf, col):
        if col is None:
            return pd.Series([''] * len(gdf), index=gdf.index)
        return gdf[col].astype(str).str.replace('\\.0$', '', regex=True).str.strip()

    dong_gdf['code'] = _norm_gdf_col_to_str(dong_gdf, dong_code_col)
    dong_gdf['name'] = _norm_gdf_col_to_str(dong_gdf, dong_name_col)

    # (CSV column detection is done later after reading location preferences)

    # helper to safely get numeric values
    def _num(val):
        try:
            if val is None or val == '' or pd.isna(val):
                return 0.0
            return float(str(val).replace(',', '').strip())
        except Exception:
            return 0.0

    # We only need 동 records that match the queried city/district/area from main.py
    # Instead of importing main (which may run heavy side-effects), parse main.py AST to extract variables safely.
    def _read_locations_from_main(root: Path):
        main_path = root / 'main.py'
        if not main_path.exists():
            return None, None, None
        try:
            import ast
            src = main_path.read_text(encoding='utf-8')
            tree = ast.parse(src)
            vals = {}
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in ('location_si', 'location_gu', 'location_dong'):
                            try:
                                vals[target.id] = ast.literal_eval(node.value)
                            except Exception:
                                # fallback for non-literal values
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

    location_si, location_gu, location_dong = _read_locations_from_main(repo_root)

    # prepare CSV columns
    code_col = _choose_col(df, ['area_code', 'area code', 'area_code', '지역코드'])
    city_col = _choose_col(df, ['city_name', 'sido', 'SIDO_NM', 'city'])
    district_col = _choose_col(df, ['district_name', 'sigungu', 'SIGUNGU_NM', 'district'])
    area_name_col = _choose_col(df, ['area_name', 'area', 'ADM_NM', 'area_name'])

    total_col = _choose_col(df, ['total_population', '총 인구', '총인구', 'to_in_001', 'in_001'])
    avg_col = _choose_col(df, ['average_age', '평균 나이', 'to_in_002'])
    dens_col = _choose_col(df, ['population_density', '인구밀도', 'to_in_003'])
    aging_col = _choose_col(df, ['aging_index', '노령화지수', 'to_in_004', 'aging_index'])
    elder_dep_col = _choose_col(df, ['elderly_dependency_ratio', 'elderly_dependency'])
    youth_dep_col = _choose_col(df, ['youth_dependency_ratio', 'youth_dependency'])
    male_col = _choose_col(df, ['total_population_male', 'male'])
    female_col = _choose_col(df, ['total_population_female', 'female'])

    age_cols = _age_group_columns(df)

    # filter merged CSV by provided location parts (require all three if provided)
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

    records = []

    # For each matched CSV row, find corresponding dong geometry and produce a single '동' record
    for _, row in matched.iterrows():
        area_code = str(row.get(code_col, '')).strip() if code_col else ''
        # find geometry in dong_gdf by code equality
        geom_row = dong_gdf[dong_gdf['code'].astype(str) == area_code]
        if len(geom_row) == 0:
            # try startswith fallback
            geom_row = dong_gdf[dong_gdf['code'].astype(str).str.startswith(area_code)] if area_code else geom_row
        if len(geom_row) == 0:
            # skip if no matching dong found
            continue

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

    # We no longer need geometry in the output GeoJSON; build a FeatureCollection with properties only
    features = []
    for rec in records:
        # properties only; no geometry key
        props = dict(rec)
        # ensure no geometry field exists
        props.pop('geometry', None)
        features.append({
            "type": "Feature",
            "properties": props
        })

    feature_collection = {
        "type": "FeatureCollection",
        "features": features
    }

    out_geojson = repo_root / 'Location_Population_Data' / 'combined_locations_population.geojson'
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    import json
    # remove existing file if present so we always create/overwrite
    try:
        out_geojson.unlink(missing_ok=True)
    except TypeError:
        # Python <3.8 fallback
        if out_geojson.exists():
            out_geojson.unlink()

    with open(out_geojson, 'w', encoding='utf-8') as fh:
        json.dump(feature_collection, fh, ensure_ascii=False, indent=2)

    # summary statistics print
    print("\n=== Summary Statistics ===")
    # Count by administrative level in the records
    total_pop_sum = 0
    level_counts = {"시도": 0, "시군구": 0, "동": 0}
    for rec in records:
        lvl = rec.get('administrative_level')
        if lvl in level_counts:
            level_counts[lvl] += 1
        total_pop_sum += int(rec.get('total_population', 0) or 0)

    print(f"시도: {level_counts['시도']}")
    print(f"시군구: {level_counts['시군구']}")
    print(f"동: {level_counts['동']}")
    print(f"\nTotal population: {total_pop_sum:,.0f}")

    # return the list of property records (no geometry GeoDataFrame)
    return records

def query_location_and_write_text(result_gdf):
    """Read location_si/gu/dong from main.py, query the GeoDataFrame and write a simple text report."""
    # Read location variables from main.py without importing it (avoid executing its code)
    def _read_locations_from_main(root: Path):
        main_path = root / 'main.py'
        if not main_path.exists():
            return None, None, None
        try:
            import ast
            src = main_path.read_text(encoding='utf-8')
            tree = ast.parse(src)
            vals = {}
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in ('location_si', 'location_gu', 'location_dong'):
                            try:
                                vals[target.id] = ast.literal_eval(node.value)
                            except Exception:
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

    location_si, location_gu, location_dong = _read_locations_from_main(Path(__file__).resolve().parents[0])
    # result_gdf is a list of property dicts (records)
    records = result_gdf if isinstance(result_gdf, list) else list(result_gdf)

    def contains_ignore_case(hay, needle):
        try:
            return needle.lower() in hay.lower()
        except Exception:
            return False

    matched = []
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
            matched.append(rec)

    out_lines = []
    out_lines.append(f"Querying for: {location_si} / {location_gu} / {location_dong}")
    if not matched:
        out_lines.append("No matching feature found in GeoJSON for requested location")
    else:
        out_lines.append(f"Found {len(matched)} matching feature(s). Showing first:")
        row = matched[0]
        # write requested fields
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

    out_path = Path(__file__).resolve().parents[0] / 'Location_Population_Data' / 'location_query_result.txt'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out_lines))

    print(f"Wrote query result to: {out_path}")


if __name__ == '__main__':
    print('Generating GeoJSON (will create or overwrite Location_Population_Data/combined_locations_population.geojson)')
    gdf = generate_geojson_from_csv_and_shapefiles()
    query_location_and_write_text(gdf)
    print('\n✓ Processing complete!')
