import geopandas as gpd
import pandas as pd
import os
from pathlib import Path
from Read_GIS_File import read_shapefile

def combine_boundary_population_data():
    
    base_path = "Dataset/경계/경계/통계청_SGIS 행정구역 통계 및 경계_20240630"
    
    # Load boundary shapefiles using Read_GIS_File.read_shapefile so we get the same read logic
    print("Loading boundary shapefiles via Read_GIS_File.read_shapefile...")
    sido_path = f"{base_path}/2. 경계/1. 2024년 2분기 기준 시도 경계/bnd_sido_00_2024_2Q.shp"
    sigungu_path = f"{base_path}/2. 경계/2. 2024년 2분기 기준 시군구 경계/bnd_sigungu_00_2024_2Q.shp"
    dong_path = f"{base_path}/2. 경계/3. 2024년 2분기 기준 행정동 경계/bnd_dong_00_2024_2Q.shp"

    def _read_and_print(shp_path: str, label: str) -> gpd.GeoDataFrame:
        """Read a shapefile using Read_GIS_File.read_shapefile and print a short summary (rows, columns, CRS, head)."""
        try:
            gdf = read_shapefile(shp_path)
        except FileNotFoundError as e:
            print(f"Shapefile not found for {label}: {shp_path}")
            raise
        except Exception as e:
            print(f"Failed reading {label} shapefile: {e}")
            raise

        # Print summary similar to Read_GIS_File.main
        print(f"\n=== {label} layer summary ===")
        print(f"Path: {shp_path}")
        print(f"Rows: {len(gdf):,}")
        print(f"Columns ({len(gdf.columns)}): {list(gdf.columns)}")
        print(f"CRS: {gdf.crs}")
        print(f"\n=== Head (first 5 rows) for {label} ===")
        try:
            print(gdf.head())
        except Exception:
            print("(unable to print head)")
        return gdf

    sido_gdf = _read_and_print(sido_path, '시도')
    sigungu_gdf = _read_and_print(sigungu_path, '시군구')
    dong_gdf = _read_and_print(dong_path, '동')

    print(f"Loaded {len(sido_gdf)} 시도, {len(sigungu_gdf)} 시군구, {len(dong_gdf)} 동 regions")
    # Prepare output directory for previews
    preview_output_dir = "Dataset/기본데이터"
    os.makedirs(preview_output_dir, exist_ok=True)

    # Print previews of the shapefile data for verification BEFORE combining with population data
    def _print_gdf_preview(gdf: gpd.GeoDataFrame, code_col: str, name_col: str, label: str, preview_n: int = 20) -> list:
        """Return list of preview lines for the GeoDataFrame (also prints them)."""
        lines = []
        lines.append(f"--- {label} preview ---")
        try:
            lines.append(f"Columns: {gdf.columns.tolist()}")
            lines.append(f"Total rows: {len(gdf)}")
            n = min(preview_n, len(gdf))
            if n == 0:
                lines.append(f"No records in {label}")
                for L in lines:
                    print(L)
                return lines
            lines.append(f"Showing first {n} rows (index, {code_col}, {name_col}, geom_type, centroid):")
            for i, r in gdf.head(n).iterrows():
                code = r.get(code_col, '')
                name = r.get(name_col, '')
                geom = r.get('geometry', None)
                geom_type = getattr(geom, 'geom_type', None)
                try:
                    if geom is not None:
                        cy, cx = geom.centroid.y, geom.centroid.x
                        lines.append(f"  idx={i} code={code} name={name} geom_type={geom_type} centroid=({cy:.6f},{cx:.6f})")
                    else:
                        lines.append(f"  idx={i} code={code} name={name} geom_type=None centroid=None")
                except Exception:
                    # Some geometries may fail centroid calculation
                    lines.append(f"  idx={i} code={code} name={name} geom_type={geom_type} centroid=unavailable")
            if len(gdf) > n:
                lines.append(f"... and {len(gdf) - n} more rows")
        except Exception as e:
            lines.append(f"Failed to preview {label}: {e}")

        # Print to console and return lines for file writing
        for L in lines:
            print(L)
        return lines

    shp_lines = []
    shp_lines += _print_gdf_preview(sido_gdf, 'SIDO_CD', 'SIDO_NM', '시도')
    shp_lines += _print_gdf_preview(sigungu_gdf, 'SIGUNGU_CD', 'SIGUNGU_NM', '시군구')
    shp_lines += _print_gdf_preview(dong_gdf, 'ADM_DR_CD', 'ADM_DR_NM', '동')

    # Write shapefile preview to a txt file for later inspection
    try:
        shp_preview_path = os.path.join(preview_output_dir, 'shapefile_preview.txt')
        with open(shp_preview_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(shp_lines))
        print(f"Wrote shapefile preview to: {shp_preview_path}")
    except Exception as e:
        print(f"Failed writing shapefile preview file: {e}")
    
    # Load population data
    print("Loading population data...")
    pop_dir = f"{base_path}/1. 통계/1. 2023년 행정구역 통계(인구)"
    
    total_pop = pd.read_csv(f"{pop_dir}/2024년기준_2023년_인구총괄(총인구).csv", encoding='cp949')
    density = pd.read_csv(f"{pop_dir}/2024년기준_2023년_인구총괄(인구밀도).csv", encoding='cp949')
    age_gender = pd.read_csv(f"{pop_dir}/2024년기준_2023년_성연령별인구.csv", encoding='cp949')
    elderly_ratio = pd.read_csv(f"{pop_dir}/2024년기준_2023년_인구총괄(노령화지수).csv", encoding='cp949')
    avg_age = pd.read_csv(f"{pop_dir}/2024년기준_2023년_인구총괄(평균나이).csv", encoding='cp949')
    
    # Print column names to understand structure
    print("\n샘플 데이터 구조:")
    print("Total Pop columns:", total_pop.columns.tolist())
    print("Total Pop shape:", total_pop.shape)
    print("Sido GDF columns:", sido_gdf.columns.tolist())
    print("\n샘플 행:")
    print(total_pop.head(3))
    
    def _norm_code(val: object) -> str:
        s = str(val)
        # remove decimal artifacts if read as float
        if "." in s:
            s = s.split(".")[0]
        return s.strip()

    # Build prefix-based lookup for multiple levels
    def build_prefix_dict(df: pd.DataFrame, value_col_idx: int = 3) -> dict[int, dict[str, float]]:
        out: dict[int, dict[str, float]] = {2: {}, 5: {}, 8: {}, 10: {}}
        if len(df.columns) <= value_col_idx:
            return out
        for _, r in df.iterrows():
            code = _norm_code(r.iloc[1])  # 행정구역코드
            try:
                value = float(r.iloc[value_col_idx]) if pd.notna(r.iloc[value_col_idx]) else 0.0
            except Exception:
                value = 0.0
            for L in (2, 5, 8, 10):
                if len(code) >= L:
                    key = code[:L]
                    out[L][key] = out[L].get(key, 0.0) + value
        return out

    pop_by_prefix = build_prefix_dict(total_pop, value_col_idx=3)
    density_by_prefix = build_prefix_dict(density, value_col_idx=3)
    elderly_by_prefix = build_prefix_dict(elderly_ratio, value_col_idx=3)
    avg_age_by_prefix = build_prefix_dict(avg_age, value_col_idx=3)

    print("\nPrefix dict sizes:")
    for L in (2, 5, 8, 10):
        print(f"  L={L}: pop={len(pop_by_prefix[L])}, dens={len(density_by_prefix[L])}")
    
    # Prepare CRS and area/centroids
    # Assume all layers share CRS; if missing, default to EPSG:4326
    for gdf in (sido_gdf, sigungu_gdf, dong_gdf):
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)

    # Precompute areas in a projected CRS (Korea 2000 / Unified CS EPSG:5179)
    try:
        sido_area = sido_gdf.to_crs(epsg=5179).area / 1_000_000
        sigungu_area = sigungu_gdf.to_crs(epsg=5179).area / 1_000_000
        dong_area = dong_gdf.to_crs(epsg=5179).area / 1_000_000
    except Exception:
        # Fallback: rough area in degrees^2 not meaningful, set to 0
        sido_area = pd.Series([0.0] * len(sido_gdf), index=sido_gdf.index)
        sigungu_area = pd.Series([0.0] * len(sigungu_gdf), index=sigungu_gdf.index)
        dong_area = pd.Series([0.0] * len(dong_gdf), index=dong_gdf.index)

    # Combine data
    combined_data = []
    
    print("\nProcessing geographic data...")
    
    # Process 시도 (Province/Metropolitan City)
    for idx, row in sido_gdf.iterrows():
        area_code = _norm_code(row.get('SIDO_CD', ''))
        area_name = row.get('SIDO_NM', 'Unknown')
        geometry = row['geometry']
        
        record = {
            'administrative_level': '시도',
            'area_code': area_code,
            'area_name': area_name,
            'total_population': pop_by_prefix[2].get(area_code[:2], 0.0),
            'population_density': density_by_prefix[2].get(area_code[:2], 0.0),
            'elderly_index': elderly_by_prefix[2].get(area_code[:2], 0.0),
            'average_age': avg_age_by_prefix[2].get(area_code[:2], 0.0),
            'geometry': geometry,
            'centroid_lat': geometry.centroid.y,
            'centroid_lon': geometry.centroid.x,
            'area_sqkm': float(sido_area.loc[idx])
        }
        combined_data.append(record)
        # Print the record as it's created for verification
        print(f"[시도] idx={idx} code={record['area_code']} name={record['area_name']} pop={record['total_population']} dens={record['population_density']} elder_idx={record['elderly_index']} avg_age={record['average_age']} centroid=({record['centroid_lat']:.6f},{record['centroid_lon']:.6f}) area_km2={record['area_sqkm']:.4f} geom_type={geometry.geom_type}")
    
    # Process 시군구 (City/County/District)
    for idx, row in sigungu_gdf.iterrows():
        area_code = _norm_code(row.get('SIGUNGU_CD', ''))
        area_name = row.get('SIGUNGU_NM', 'Unknown')
        geometry = row['geometry']
        
        record = {
            'administrative_level': '시군구',
            'area_code': area_code,
            'area_name': area_name,
            'total_population': pop_by_prefix[5].get(area_code[:5], 0.0),
            'population_density': density_by_prefix[5].get(area_code[:5], 0.0),
            'elderly_index': elderly_by_prefix[5].get(area_code[:5], 0.0),
            'average_age': avg_age_by_prefix[5].get(area_code[:5], 0.0),
            'geometry': geometry,
            'centroid_lat': geometry.centroid.y,
            'centroid_lon': geometry.centroid.x,
            'area_sqkm': float(sigungu_area.loc[idx])
        }
        combined_data.append(record)
        # Print the record as it's created for verification
        print(f"[시군구] idx={idx} code={record['area_code']} name={record['area_name']} pop={record['total_population']} dens={record['population_density']} elder_idx={record['elderly_index']} avg_age={record['average_age']} centroid=({record['centroid_lat']:.6f},{record['centroid_lon']:.6f}) area_km2={record['area_sqkm']:.4f} geom_type={geometry.geom_type}")
    
    # Process 동 (Administrative Dong)
    for idx, row in dong_gdf.iterrows():
        area_code = _norm_code(row.get('ADM_DR_CD', ''))
        area_name = row.get('ADM_DR_NM', 'Unknown')
        geometry = row['geometry']
        # Prefer 10-digit match, fallback to 8-digit aggregation
        pop_val = pop_by_prefix[10].get(area_code[:10], None)
        if pop_val is None:
            pop_val = pop_by_prefix[8].get(area_code[:8], 0.0)
        dens_val = density_by_prefix[10].get(area_code[:10], None)
        if dens_val is None:
            dens_val = density_by_prefix[8].get(area_code[:8], 0.0)
        eld_val = elderly_by_prefix[10].get(area_code[:10], None)
        if eld_val is None:
            eld_val = elderly_by_prefix[8].get(area_code[:8], 0.0)
        avg_val = avg_age_by_prefix[10].get(area_code[:10], None)
        if avg_val is None:
            avg_val = avg_age_by_prefix[8].get(area_code[:8], 0.0)

        record = {
            'administrative_level': '동',
            'area_code': area_code,
            'area_name': area_name,
            'total_population': pop_val,
            'population_density': dens_val,
            'elderly_index': eld_val,
            'average_age': avg_val,
            'geometry': geometry,
            'centroid_lat': geometry.centroid.y,
            'centroid_lon': geometry.centroid.x,
            'area_sqkm': float(dong_area.loc[idx])
        }
        combined_data.append(record)
        # Print the record as it's created for verification
        print(f"[동] idx={idx} code={record['area_code']} name={record['area_name']} pop={record['total_population']} dens={record['population_density']} elder_idx={record['elderly_index']} avg_age={record['average_age']} centroid=({record['centroid_lat']:.6f},{record['centroid_lon']:.6f}) area_km2={record['area_sqkm']:.4f} geom_type={geometry.geom_type}")
    
    result_gdf = gpd.GeoDataFrame(combined_data, geometry='geometry')
    # Carry CRS from one of the inputs (default to EPSG:4326)
    crs_to_set = sido_gdf.crs or sigungu_gdf.crs or dong_gdf.crs or "EPSG:4326"
    try:
        result_gdf.set_crs(crs_to_set, inplace=True, allow_override=True)
    except Exception:
        pass
    
    print(f"\nCreated combined dataset with {len(result_gdf)} regions")
    
    # Create output directory if it doesn't exist
    output_dir = "Dataset/기본데이터"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV (without geometry column)
    csv_df = result_gdf.drop('geometry', axis=1)
    csv_path = f"{output_dir}/지역별_인구_경계_데이터.csv"
    # Print CSV rows line-by-line for verification before saving
    print("\n--- CSV preview: printing each row to verify values before writing ---")
    # Build preview lines and also write to file
    csv_preview_lines = []
    header = ','.join(csv_df.columns.tolist())
    print(header)
    csv_preview_lines.append(header)
    for i, row in csv_df.iterrows():
        # Convert NaNs to empty strings for clearer print
        values = [('' if pd.isna(v) else str(v)) for v in row.tolist()]
        line = ','.join(values)
        print(line)
        csv_preview_lines.append(line)

    # Write CSV preview to a txt file
    try:
        csv_preview_path = os.path.join(output_dir, 'csv_preview.txt')
        with open(csv_preview_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(csv_preview_lines))
        print(f"Wrote CSV preview to: {csv_preview_path}")
    except Exception as e:
        print(f"Failed writing CSV preview file: {e}")

    csv_df.to_csv(csv_path, index=False, encoding='cp949')
    print(f"Saved CSV file to: {csv_path}")
    
    # Save as GeoJSON for spatial queries
    geojson_path = f"{output_dir}/지역별_인구_경계_데이터.geojson"
    # Allow large GeoJSON objects when writing
    os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")
    result_gdf.to_file(geojson_path, driver='GeoJSON', encoding='cp949')
    print(f"Saved GeoJSON file to: {geojson_path}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total regions: {len(result_gdf)}")
    print(f"시도: {len(result_gdf[result_gdf['administrative_level'] == '시도'])}")
    print(f"시군구: {len(result_gdf[result_gdf['administrative_level'] == '시군구'])}")
    print(f"동: {len(result_gdf[result_gdf['administrative_level'] == '동'])}")
    print(f"\nTotal population: {result_gdf['total_population'].sum():,.0f}")
    print(f"Average density: {result_gdf['population_density'].mean():.2f} people/km²")
    
    return result_gdf

if __name__ == "__main__":
    print("Starting geospatial data processing...")
    df = combine_boundary_population_data()
    print("\n✓ Processing complete!")
