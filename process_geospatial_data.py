"""
Process geospatial boundary and population data for RAG integration
Combines shapefiles with population statistics to create searchable documents
"""
import geopandas as gpd
import pandas as pd
import os
from pathlib import Path

def combine_boundary_population_data():
    """
    Combine shapefile boundary data with population statistics
    Returns a comprehensive DataFrame with geometry and demographics
    """
    
    base_path = "Dataset/경계/경계/통계청_SGIS 행정구역 통계 및 경계_20240630"
    
    # Load boundary shapefiles
    print("Loading boundary shapefiles...")
    sido_path = f"{base_path}/2. 경계/1. 2024년 2분기 기준 시도 경계/bnd_sido_00_2024_2Q.shp"
    sigungu_path = f"{base_path}/2. 경계/2. 2024년 2분기 기준 시군구 경계/bnd_sigungu_00_2024_2Q.shp"
    dong_path = f"{base_path}/2. 경계/3. 2024년 2분기 기준 행정동 경계/bnd_dong_00_2024_2Q.shp"
    
    sido_gdf = gpd.read_file(sido_path, encoding='cp949')
    sigungu_gdf = gpd.read_file(sigungu_path, encoding='cp949')
    dong_gdf = gpd.read_file(dong_path, encoding='cp949')
    
    print(f"Loaded {len(sido_gdf)} 시도, {len(sigungu_gdf)} 시군구, {len(dong_gdf)} 동 regions")
    
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
    
    # Prepare population lookup dictionaries
    # The CSV structure is: 기준년도, 행정구역코드, 통계항목, 통계값
    pop_dict = {}
    density_dict = {}
    elderly_dict = {}
    avg_age_dict = {}
    
    # Column 1 is area code (행정구역코드), Column 3 is value (통계값)
    if len(total_pop.columns) >= 4:
        for idx, row in total_pop.iterrows():
            code = str(row.iloc[1])  # 행정구역코드
            value = row.iloc[3] if pd.notna(row.iloc[3]) else 0  # 통계값
            # Aggregate values for the same code
            if code in pop_dict:
                pop_dict[code] += value
            else:
                pop_dict[code] = value
    
    if len(density.columns) >= 4:
        for idx, row in density.iterrows():
            code = str(row.iloc[1])
            value = row.iloc[3] if pd.notna(row.iloc[3]) else 0
            density_dict[code] = value
    
    if len(elderly_ratio.columns) >= 4:
        for idx, row in elderly_ratio.iterrows():
            code = str(row.iloc[1])
            value = row.iloc[3] if pd.notna(row.iloc[3]) else 0
            elderly_dict[code] = value
    
    if len(avg_age.columns) >= 4:
        for idx, row in avg_age.iterrows():
            code = str(row.iloc[1])
            value = row.iloc[3] if pd.notna(row.iloc[3]) else 0
            avg_age_dict[code] = value
    
    print(f"\nPopulation dictionary has {len(pop_dict)} entries")
    print(f"Sample codes from pop_dict:", list(pop_dict.keys())[:5])
    print(f"Sample SIDO codes from shapefile:", sido_gdf['SIDO_CD'].head().tolist())
    
    # Combine data
    combined_data = []
    
    print("\nProcessing geographic data...")
    
    # Process 시도 (Province/Metropolitan City)
    for idx, row in sido_gdf.iterrows():
        area_code = str(row.get('SIDO_CD', ''))
        area_name = row.get('SIDO_NM', 'Unknown')
        geometry = row['geometry']
        
        combined_data.append({
            'administrative_level': '시도',
            'area_code': area_code,
            'area_name': area_name,
            'total_population': pop_dict.get(area_code, 0),
            'population_density': density_dict.get(area_code, 0),
            'elderly_index': elderly_dict.get(area_code, 0),
            'average_age': avg_age_dict.get(area_code, 0),
            'geometry': geometry,
            'centroid_lat': geometry.centroid.y,
            'centroid_lon': geometry.centroid.x,
            'area_sqkm': geometry.area / 1000000  # Convert to km²
        })
    
    # Process 시군구 (City/County/District)
    for idx, row in sigungu_gdf.iterrows():
        area_code = str(row.get('SIGUNGU_CD', ''))
        area_name = row.get('SIGUNGU_NM', 'Unknown')
        geometry = row['geometry']
        
        combined_data.append({
            'administrative_level': '시군구',
            'area_code': area_code,
            'area_name': area_name,
            'total_population': pop_dict.get(area_code, 0),
            'population_density': density_dict.get(area_code, 0),
            'elderly_index': elderly_dict.get(area_code, 0),
            'average_age': avg_age_dict.get(area_code, 0),
            'geometry': geometry,
            'centroid_lat': geometry.centroid.y,
            'centroid_lon': geometry.centroid.x,
            'area_sqkm': geometry.area / 1000000
        })
    
    # Process 동 (Administrative Dong)
    for idx, row in dong_gdf.iterrows():
        area_code = str(row.get('ADM_DR_CD', ''))
        area_name = row.get('ADM_DR_NM', 'Unknown')
        geometry = row['geometry']
        
        combined_data.append({
            'administrative_level': '동',
            'area_code': area_code,
            'area_name': area_name,
            'total_population': pop_dict.get(area_code, 0),
            'population_density': density_dict.get(area_code, 0),
            'elderly_index': elderly_dict.get(area_code, 0),
            'average_age': avg_age_dict.get(area_code, 0),
            'geometry': geometry,
            'centroid_lat': geometry.centroid.y,
            'centroid_lon': geometry.centroid.x,
            'area_sqkm': geometry.area / 1000000
        })
    
    result_gdf = gpd.GeoDataFrame(combined_data, geometry='geometry')
    
    print(f"\nCreated combined dataset with {len(result_gdf)} regions")
    
    # Create output directory if it doesn't exist
    output_dir = "Dataset/기본데이터"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to Excel (without geometry column)
    excel_df = result_gdf.drop('geometry', axis=1)
    excel_path = f"{output_dir}/지역별_인구_경계_데이터.xlsx"
    excel_df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"Saved Excel file to: {excel_path}")
    
    # Save as GeoJSON for spatial queries
    geojson_path = f"{output_dir}/지역별_인구_경계_데이터.geojson"
    result_gdf.to_file(geojson_path, driver='GeoJSON', encoding='utf-8')
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
