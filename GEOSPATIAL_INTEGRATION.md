# Geospatial Data Integration for RAG System

## Overview
This document explains how geographical boundary data and population statistics have been integrated into the disaster management RAG system.

## What Data Was Added

### 1. Boundary Data (Shapefiles)
- **시도 (Province/Metropolitan)**: 17 regions
- **시군구 (City/County/District)**: 252 regions  
- **동 (Administrative Dong)**: 3,558 regions
- Total: **3,827 administrative regions** across South Korea

### 2. Population Statistics (CSV)
- Total population by area
- Population density (people/km²)
- Elderly index (노령화지수)
- Average age
- Gender and age distribution

## Integration Architecture

### Data Processing Flow
```
Shapefiles (경계 데이터)     Population CSVs (인구 통계)
        ↓                              ↓
        └──────────────┬───────────────┘
                       ↓
          process_geospatial_data.py
                       ↓
        ┌──────────────┴───────────────┐
        ↓                              ↓
   GeoJSON File                  Excel File
지역별_인구_경계_데이터.geojson  지역별_인구_경계_데이터.xlsx
        ↓
knowledge_base_copy1.py
  (load_geospatial_documents)
        ↓
   Vector Database (FAISS)
        ↓
   RAG Retrieval Pipeline
```

### File Changes

#### 1. `process_geospatial_data.py` (NEW)
- Reads shapefiles from `Dataset/경계/경계/통계청_SGIS 행정구역 통계 및 경계_20240630/`
- Reads population CSVs from same directory
- Combines boundary geometry with demographic data
- Outputs:
  - `Dataset/기본데이터/지역별_인구_경계_데이터.geojson` (for spatial queries)
  - `Dataset/기본데이터/지역별_인구_경계_데이터.xlsx` (for human review)

#### 2. `knowledge_base_copy1.py` (MODIFIED)
- Added `import geopandas as gpd`
- Added `load_geospatial_documents()` function:
  - Loads GeoJSON data
  - Creates rich text descriptions for each region including:
    - Administrative level, area code, area name
    - Total population, density, area size
    - GPS coordinates
    - Risk analysis (high density, large population, elderly concerns)
    - Disaster response considerations
  - Returns Document objects with metadata
- Modified `build_vectorstores()`:
  - Calls `load_geospatial_documents()`
  - Adds geospatial docs to `basic_docs`
  - All documents are embedded and stored in FAISS vector DB

#### 3. `nodes.py` (MODIFIED)
- Modified `retrieval_basic_node()`:
  - Increased k from 2 to 3 for more coverage
  - Separates geospatial and non-geospatial documents
  - If no geospatial docs found, does additional search with location terms
  - Ensures geographic context is included in retrieval
- Modified LLM prompt in `llm_node()`:
  - Added explicit instructions to consider geographic characteristics
  - Emphasizes population density, population size, elderly index
  - Requests scaled impact assessment based on regional data
  - Asks for resource allocation based on population

#### 4. `requirements.txt` (MODIFIED)
Added geospatial processing libraries:
```
geopandas
openpyxl
shapely
fiona
pyproj
```

#### 5. `test_geo_integration.py` (NEW)
- Tests vector store building with geospatial data
- Validates geospatial document retrieval
- Provides framework for full pipeline testing

## How It Works

### Document Structure
Each geographic region becomes a searchable document:

```
지역명: 서울특별시 강남구
행정구역 수준: 시군구
지역코드: 11680
총 인구: 542,000명
인구 밀도: 13,845.3명/km²
면적: 39.15km²
중심 좌표: (위도 37.5172, 경도 127.0473)
평균 연령: 42.3세
노령화지수: 95.2

재난 위험도 분석:
- 인구밀도 위험도: 높음
- 인구 규모: 대규모
- 고령인구 관리 필요: 아니오
- 대피 소요 예상 시간: 장시간 소요 (고밀도)

재난 대응 시 고려사항:
- 고밀도 지역으로 신속한 대피가 어려울 수 있음
- 대규모 인구로 인해 대량의 대피 및 구호 자원 필요
```

### Retrieval Process
1. User query: "서울 강남구에서 태풍 발생 시 예상 피해는?"
2. `retrieval_basic_node` searches vector DB
3. Finds relevant geospatial documents (Gangnam-gu data)
4. LLM receives context including:
   - Population: 542,000 people
   - Density: 13,845 people/km²
   - Area: 39.15 km²
   - Risk factors: High density, large population
5. LLM generates response with location-specific recommendations

### Benefits

1. **Location-Aware Risk Assessment**
   - High-density areas flagged for evacuation challenges
   - Large populations trigger resource scaling alerts
   - Elderly populations get special consideration

2. **Resource Planning**
   - Evacuation shelter capacity based on population
   - Emergency supplies scaled to area size
   - Medical resources adjusted for demographic profile

3. **Jurisdiction Clarity**
   - Administrative codes link to responsible authorities
   - Multi-level (시도/시군구/동) granularity

4. **Historical Context**
   - Can correlate past disasters with specific regions
   - Pattern detection for vulnerable areas

## Usage Examples

### Running the Integration

1. **Process the geospatial data** (first time only):
```bash
cd /home/sslab/Documents/Nishtha/RAG_system
python process_geospatial_data.py
```

2. **Test the integration**:
```bash
python test_geo_integration.py
```

3. **Run the full RAG system**:
```bash
python main.py
```

### Query Examples

**Without geographic data (before):**
- Query: "태풍 발생 시 대응 방안"
- Response: Generic typhoon response procedures

**With geographic data (now):**
- Query: "서울 강남구에서 태풍 발생 시 대응 방안"
- Response: 
  - Gangnam-gu specific considerations
  - Population: 542,000 people need evacuation planning
  - High density (13,845/km²) means slow evacuation
  - Requires 50+ emergency shelters
  - Need 1000+ emergency personnel

## Technical Details

### Metadata Structure
Each geospatial document includes:
```python
{
    "source": "지역별_인구_경계_데이터",
    "area_code": "11680",
    "area_name": "강남구",
    "administrative_level": "시군구",
    "population": 542000,
    "density": 13845.3,
    "type": "geospatial"
}
```

### Risk Classification
- **Density Risk**: 
  - 매우 높음: > 15,000/km²
  - 높음: 10,000-15,000/km²
  - 중간: 5,000-10,000/km²
  - 낮음: < 5,000/km²

- **Population Scale**:
  - 대규모: > 500,000
  - 중규모: 100,000-500,000
  - 소규모: 10,000-100,000
  - 미소규모: < 10,000

- **Elderly Concern**: 노령화지수 > 100

## Troubleshooting

### Issue: No geospatial documents found
**Solution**: 
1. Ensure `지역별_인구_경계_데이터.geojson` exists in `Dataset/기본데이터/`
2. Run `process_geospatial_data.py` if missing
3. Check file permissions

### Issue: Encoding errors with Korean characters
**Solution**: 
- Shapefiles use CP949 encoding (handled automatically by geopandas)
- Output files use UTF-8

### Issue: Population data is 0
**Solution**:
- CSV structure may have changed
- Check column indices in `process_geospatial_data.py`
- Verify area code matching between shapefiles and CSVs

## Future Enhancements

1. **Spatial Queries**: Add actual geographic filtering (within radius, boundary intersection)
2. **Real-time Data**: Update with current population estimates
3. **Infrastructure Layer**: Add hospitals, fire stations, shelters
4. **Historical Disasters**: Overlay past disaster locations with current demographics
5. **Visualization**: Generate maps showing risk zones

## Dependencies

```
geopandas==1.0.1      # Geographic data processing
shapely==2.0.4        # Geometric operations
fiona==1.10.1         # Shapefile reading
pyproj==3.6.1         # Coordinate transformations
openpyxl==3.1.5       # Excel file output
pandas>=2.0.0         # Data manipulation
```

## References

- Data Source: 통계청 SGIS (Statistics Korea)
- Boundary Data: 2024 Q2 administrative boundaries
- Population Data: 2023 population statistics (2024 standard)
- Coordinate System: WGS84 (EPSG:4326)
