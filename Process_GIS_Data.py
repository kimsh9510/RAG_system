import subprocess
import sys
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
    cols = [c for c in df.columns if c.startswith('age_group_')]
    # sort by numeric suffix if possible
    def key(c):
        try:
            return int(c.split('_')[-1])
        except Exception:
            return c
    return sorted(cols, key=key)


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
    """Run Combine_Loca_Popu_Data.py, read the merged CSV and boundary shapefiles,
    join attributes and write a single GeoJSON for downstream use.
    """
    repo_root = Path(__file__).resolve().parents[0]

    # Ensure merged CSV exists by running Combine_Loca_Popu_Data.py (it may be a no-op if CSV exists)
    try:
        subprocess.run([sys.executable, str(repo_root / 'Combine_Loca_Popu_Data.py')], check=False)
    except Exception:
        # continue even if running fails; we'll try to read the CSV
        pass

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

    # helper to safely get numeric values
    def _num(val):
        try:
            if val is None or val == '' or pd.isna(val):
                return 0.0
            return float(str(val).replace(',', '').strip())
        except Exception:
            return 0.0

    # We only need 동 records that match the queried city/district/area from main.py
    try:
        import main as main_mod
        location_si = getattr(main_mod, 'location_si', None)
        location_gu = getattr(main_mod, 'location_gu', None)
        location_dong = getattr(main_mod, 'location_dong', None)
    except Exception:
        location_si = location_gu = location_dong = None

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
            # skip if no geometry
            continue
        geom = geom_row.iloc[0].geometry

        city_val = row.get(city_col, '') if city_col else ''
        district_val = row.get(district_col, '') if district_col else ''
        area_val = row.get(area_name_col, '') if area_name_col else ''

        rec = {
            'administrative_level': '동',
            'area_code': area_code,
            'area_name': f"{city_val} / {district_val} / {area_val}",
            'total_population': int(_num(row[total_col])) if total_col and total_col in row.index else 0,
            'average_age': float(_num(row[avg_col])) if avg_col and avg_col in row.index else 0.0,
            'population_density': float(_num(row[dens_col])) if dens_col and dens_col in row.index else 0.0,
            'aging_index': float(_num(row[aging_col])) if aging_col and aging_col in row.index else 0.0,
            'elderly_dependency_ratio': float(_num(row[elder_dep_col])) if elder_dep_col and elder_dep_col in row.index else 0.0,
            'youth_dependency_ratio': float(_num(row[youth_dep_col])) if youth_dep_col and youth_dep_col in row.index else 0.0,
            'total_population_male': int(_num(row[male_col])) if male_col and male_col in row.index else 0,
            'total_population_female': int(_num(row[female_col])) if female_col and female_col in row.index else 0,
            'geometry': geom
        }
        records.append(rec)

    result_gdf = gpd.GeoDataFrame(records, geometry='geometry')
    try:
        result_gdf.set_crs(dong_gdf.crs, inplace=True)
    except Exception:
        pass

    out_geojson = repo_root / 'Location_Population_Data' / 'combined_locations_population.geojson'
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    # write geojson (contains only matched 동 features)
    result_gdf.to_file(str(out_geojson), driver='GeoJSON', encoding='utf-8')

    # summary statistics print
    print("\n=== Summary Statistics ===")
    # we only created 동-level features here
    print(f"시도: {len(result_gdf[result_gdf['administrative_level'] == '시도'])}")
    print(f"시군구: {len(result_gdf[result_gdf['administrative_level'] == '시군구'])}")
    print(f"동: {len(result_gdf[result_gdf['administrative_level'] == '동'])}")
    print(f"\nTotal population: {result_gdf['total_population'].sum():,.0f}")

    return result_gdf


def query_location_and_write_text(result_gdf):
    """Read location_si/gu/dong from main.py, query the GeoDataFrame and write a simple text report."""
    try:
        import main as main_mod
        location_si = getattr(main_mod, 'location_si', None)
        location_gu = getattr(main_mod, 'location_gu', None)
        location_dong = getattr(main_mod, 'location_dong', None)
    except Exception:
        location_si = location_gu = location_dong = None

    matched = result_gdf
    try:
        if location_si:
            matched = matched[matched['area_name'].astype(str).str.contains(str(location_si), na=False, case=False)]
        if location_gu:
            matched = matched[matched['area_name'].astype(str).str.contains(str(location_gu), na=False, case=False)]
        if location_dong:
            matched = matched[matched['area_name'].astype(str).str.contains(str(location_dong), na=False, case=False)]
    except Exception:
        matched = result_gdf

    out_lines = []
    out_lines.append(f"Querying for: {location_si} / {location_gu} / {location_dong}")
    if matched is None or len(matched) == 0:
        out_lines.append("No matching feature found in GeoJSON for requested location")
    else:
        out_lines.append(f"Found {len(matched)} matching feature(s). Showing first:")
        row = matched.iloc[0]
        # write requested fields
        out_lines.append(f"administrative_level: {row['administrative_level']}")
        out_lines.append(f"area_code: {row['area_code']}")
        out_lines.append(f"area_name: {row['area_name']}")
        out_lines.append(f"age_wise_population: {row['age_wise_population']}")
        out_lines.append(f"total_population: {row['total_population']}")
        out_lines.append(f"average_age: {row['average_age']}")
        out_lines.append(f"population_density: {row['population_density']}")
        out_lines.append(f"aging_index: {row['aging_index']}")
        out_lines.append(f"elderly_dependency_ratio: {row['elderly_dependency_ratio']}")
        out_lines.append(f"youth_dependency_ratio: {row['youth_dependency_ratio']}")
        out_lines.append(f"total_population_male: {row['total_population_male']}")
        out_lines.append(f"total_population_female: {row['total_population_female']}")

    out_path = Path(__file__).resolve().parents[0] / 'Location_Population_Data' / 'location_query_result.txt'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out_lines))

    print(f"Wrote query result to: {out_path}")


if __name__ == '__main__':
    print('Running Combine_Loca_Popu_Data.py (if needed) and generating GeoJSON...')
    gdf = generate_geojson_from_csv_and_shapefiles()
    query_location_and_write_text(gdf)
    print('\n✓ Processing complete!')
