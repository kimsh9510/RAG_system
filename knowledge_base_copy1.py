#벡터 DB구축 - > 지식그래프 DB로 변경 예정
from PyPDF2 import PdfReader
import pdfplumber
import logging
import warnings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models import load_embeddings
import glob
import docx # python-docx
import os
import zipfile #hwpx
import xml.etree.ElementTree as ET #hwpx
import pandas as pd
import os
import geopandas as gpd
import sys
import re
import logging
import warnings
from datetime import datetime
from pathlib import Path

def load_all_documents_to_list(directory_path):
    all_documents = glob.glob(os.path.join(directory_path,"*"))
    
    documents=[]

    for file_path in all_documents:
        try:
            # skip directories (new behavior)
            if os.path.isdir(file_path):
                print(f"폴더 건너뜀: {file_path}")
                continue

            if file_path.endswith(".pdf"): #확장자 pdf로 끝나는 경우 
                # prefer pdfplumber for better text + table extraction; fallback to PyPDF2
                # reduce noisy decompression / pdfminer logs which often flood console for damaged PDFs
                logging.getLogger("pdfminer").setLevel(logging.ERROR)
                logging.getLogger("pdfplumber").setLevel(logging.ERROR)
                warnings.filterwarnings("ignore", category=UserWarning, module=r"pdfplumber")

                try:
                    with pdfplumber.open(file_path) as pdf:
                        for page_num, page in enumerate(pdf.pages):
                            text = page.extract_text() or ""
                            # extract tables as text as well
                            tables = page.extract_tables()
                            table_text = ""
                            if tables:
                                for table in tables:
                                    table_text += "\n\n--- TABLE START ---\n"
                                    for row in table:
                                        table_text += " | ".join(map(str, row)) + "\n"
                                    table_text += "--- TABLE END ---\n\n"
                            full_content = text + table_text
                            # normalize and split very long runs to avoid oversized chunks later
                            full_content = _normalize_and_break_long_tokens(full_content)
                            documents.append(Document(page_content=full_content, metadata={"source": file_path, "page": page_num}))
                except Exception:
                    # fallback
                    reader=PdfReader(file_path)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text() or ""
                        text = _normalize_and_break_long_tokens(text)
                        documents.append(Document(page_content=text, metadata={"source": file_path, "page": page_num}))

            elif file_path.endswith(".docx"): #docx로 끝나는 경우
                doc = docx.Document(file_path)
                full_text = "\n".join([para.text for para in doc.paragraphs])
                full_text = _normalize_and_break_long_tokens(full_text)
                documents.append(Document(page_content=full_text,metadata={"source": file_path}))

            elif file_path.endswith(".hwpx"):
                full_text=""
                with zipfile.ZipFile(file_path,'r') as z:
                    section_xml_files = [f for f in z.namelist() if f.startswith('Contents/section') and f.endswith('.xml')]
                    for section_file in sorted(section_xml_files):
                        xml_content = z.read(section_file)
                        root = ET.fromstring(xml_content)
                        for text_element in root.iter('{http://www.hancom.co.kr/hwpml/2011/paragraph}t'):
                            if text_element.text:
                                full_text += text_element.text + "\n"
                full_text = _normalize_and_break_long_tokens(full_text)
                documents.append(Document(page_content=full_text, metadata={"source":file_path}))

            elif file_path.endswith(".txt"): # 확장자가 .txt로 끝나는 경우
                # 'r' 모드(읽기 전용), 'utf-8' 인코딩으로 파일을 엽니다.
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read() 
                    # remove lines that contain the word 사진 (photos), they are noisy in some datasets
                    pattern = r'^.*사진.*$\n?'
                    full_text = re.sub(pattern, '', raw_text, flags=re.MULTILINE)
                    full_text = _normalize_and_break_long_tokens(full_text)
                documents.append(
                    Document(
                        page_content=full_text, 
                        metadata={"source": os.path.basename(file_path)}
                    )
                )
                """ 읽어올 데이터 
                1. 1행 G열
                2. 2행~끝까지 F(유형),G(세부사항),H(주소),O(최종소요시간),V(첫 조치 소요시간) 행 
                """
                #1행 G열 데이터 읽기 (딕셔너리 형태로 받아옴)
                df1 = pd.read_excel(file_path, sheet_name=0,header=None, nrows=1)  # 첫 번째 시트만 읽기, 첫행 데이터로 취급, 1행의 G열만 필요 : 1행만 읽기 
                g_value = df1.iloc[0,6]
                parse_data_g_value = parse_multiline_cell(g_value)
                header_content = ", ".join([f"{k}: {v}" for k, v in parse_data_g_value.items()])
                doc = Document(
                    page_content=f"문서 요약 정보: {header_content}",
                    metadata={"source": os.path.basename(file_path), "section": "header"}
                )
                documents.append(doc)

                #2행부터 끝까지 읽어옴
                df2= pd.read_excel(file_path, sheet_name=0, header=1, usecols="F,G,H,O,V")
                for index, row in df2.iterrows():
                    row_content = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    doc = Document(
                        page_content=row_content,
                        metadata={
                            "source": os.path.basename(file_path),
                            "row": index + 3 # 엑셀 실제 행 번호와 맞춤
                        }
                    )
                    documents.append(doc)

        except Exception as e:
            # 해당 파일은 건너뛰고 오류 메시지를 출력
            print(f"Error processing {file_path}: {e}")
    # don't print large documents list in normal runs; keep a short summary
    print(f"Loaded {len(documents)} raw documents from {directory_path}")
    return documents


def _normalize_and_break_long_tokens(text: str, max_run: int = 400, break_size: int = 200) -> str:
    """Normalize whitespace and break extremely long non-space runs which cause the
    CharacterTextSplitter to emit warnings like 'Created a chunk of size X, which is longer than the specified Y'.

    - max_run: consider runs of non-whitespace longer than this as 'too long'
    - break_size: break those runs into pieces of length break_size separated by spaces
    """
    if not isinstance(text, str) or not text:
        return text

    # normalize whitespace (collapse many newlines/spaces to single spaces, but keep newlines for splitting)
    # preserve table markers/newlines: replace multiple spaces but keep line breaks
    # first, replace Windows line endings
    t = text.replace('\r\n', '\n').replace('\r', '\n')
    # collapse repeated spaces/tabs but keep single newlines
    t = re.sub(r"[ \t]+", " ", t)

    # break extremely long continuous non-space sequences (e.g., corrupted pdf streams, long table rows)
    def _breaker(m):
        s = m.group(0)
        parts = [s[i:i+break_size] for i in range(0, len(s), break_size)]
        return " ".join(parts)

    t = re.sub(rf"\S{{{max_run},}}", _breaker, t)
    return t

def parse_multiline_cell(cell_text):
    """
    "키: 값" 형태의 여러 줄 문자열을 딕셔너리로 변환
    """
    if not isinstance(cell_text, str):
        return {}
        
    data_dict = {}
    lines = cell_text.splitlines() #\n 기준 나누기
    
    for line in lines:
        # 2. 각 줄을 ':' 기준으로 -> ['유형', '연쇄사항']
        if ':' in line:
            key, value = line.split(':', 1)
            data_dict[key.strip()] = value.strip()
            
    return data_dict


def load_geospatial_documents():
    """
    Load geospatial data and convert to documents for RAG
    Returns a list of Document objects with geographic and population information
    """
    documents = []
    
    try:
        # Read query context from the running main module (__main__) first, then env vars, then defaults
        main_mod = sys.modules.get("__main__")
        main_si = getattr(main_mod, "location_si", None) if main_mod else None
        main_gu = getattr(main_mod, "location_gu", None) if main_mod else None
        main_dong = getattr(main_mod, "location_dong", None) if main_mod else None
        main_disaster = getattr(main_mod, "disaster", None) if main_mod else None

        location_si = main_si or os.environ.get("LOCATION_SI") or "서울시"
        location_gu = main_gu or os.environ.get("LOCATION_GU") or "서초구"
        location_dong = main_dong or os.environ.get("LOCATION_DONG") or "서초동"
        disaster = main_disaster or os.environ.get("DISASTER") or "재난"

        # We'll collect output into a buffer and write to a timestamped file instead of printing
        out_lines = []
        out_lines.append(f"Query context -> 재난: {disaster}, 행정: {location_si} {location_gu} {location_dong}")

        # Load the combined geospatial data
        geojson_path = "Dataset/기본데이터/지역별_인구_경계_데이터.geojson"

        if not os.path.exists(geojson_path):
            print(f"Warning: Geospatial data not found at {geojson_path}")
            print("Please run process_geospatial_data.py first to generate the data")
            return documents

        # Load the geojson silently; status will be written to the output file
        # Allow large GeoJSON features (remove size limit)
        os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")
        # Try pyogrio (default) first, then fall back to fiona engine for robustness
        try:
            gdf = gpd.read_file(geojson_path)
        except Exception as e1:
            print(f"pyogrio read failed ({e1}); retrying with fiona engine...")
            gdf = gpd.read_file(geojson_path, engine="fiona")

        # Prepare location parts and normalize
        parts = [p for p in (location_si, location_gu, location_dong) if p]
        parts_norm = [p.strip().lower() for p in parts]

        # Choose candidate fields to search for location strings
        candidate_fields = ['area_name', 'administrative_level', 'area_code']
        search_fields = [f for f in candidate_fields if f in gdf.columns]
        if not search_fields:
            search_fields = list(gdf.columns)

        def _norm_val(v):
            try:
                return str(v).lower()
            except Exception:
                return ""

        # Filter rows where every location part appears somewhere in the searchable text
        mask = []
        for idx, row in gdf.iterrows():
            row_text = " ".join([_norm_val(row.get(f, "")) for f in search_fields])
            matched = all(p in row_text for p in parts_norm)
            mask.append(matched)
        gdf_matches = gdf[mask]

        match_count = len(gdf_matches)
        out_lines.append(f"Found {match_count} geospatial row(s) matching the query location.")

        if match_count == 0:
            # write a short file noting zero matches and return
            out_dir = Path("outputs") / "geospatial_queries"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = re.sub(r"\s+", "_", f"{location_si}_{location_gu}_{location_dong}_{disaster}")
            out_path = out_dir / f"{ts}_{safe_name}_no_matches.txt"
            out_lines.append("No geospatial rows matched the provided location.")
            out_path.write_text("\n".join(out_lines), encoding="utf-8")
            return documents

        # For each matched row, create Document and print a short summary
        for idx, row in gdf_matches.iterrows():
            density = float(row.get('population_density', 0) or 0)
            population = float(row.get('total_population', 0) or 0)
            elderly_index = float(row.get('elderly_index', 0) or 0)
            area_name = _norm_val(row.get('area_name', ''))
            administrative_level = _norm_val(row.get('administrative_level', ''))
            area_code = _norm_val(row.get('area_code', ''))
            area_sqkm = float(row.get('area_sqkm', 0) or 0)
            centroid_lat = float(row.get('centroid_lat', 0) or 0)
            centroid_lon = float(row.get('centroid_lon', 0) or 0)

            if density > 15000:
                density_risk = "매우 높음"
            elif density > 10000:
                density_risk = "높음"
            elif density > 5000:
                density_risk = "중간"
            else:
                density_risk = "낮음"
            
            if population > 500000:
                pop_risk = "대규모"
            elif population > 100000:
                pop_risk = "중규모"
            elif population > 10000:
                pop_risk = "소규모"
            else:
                pop_risk = "미소규모"
            
            elderly_concern = "예" if elderly_index > 100 else "아니오"

            content = f"""
지역명: {row.get('area_name', '')}
행정구역 수준: {row.get('administrative_level', '')}
지역코드: {row.get('area_code', '')}
총 인구: {int(population):,}명
인구 밀도: {density:.1f}명/km²
면적: {area_sqkm:.2f}km²
중심 좌표: (위도 {centroid_lat:.4f}, 경도 {centroid_lon:.4f})
평균 연령: {row.get('average_age', 0):.1f}세
노령화지수: {elderly_index:.1f}

재난 위험도 분석:
- 인구밀도 위험도: {density_risk}
- 인구 규모: {pop_risk}
- 고령인구 관리 필요: {elderly_concern}
- 대피 소요 예상 시간: {('짧음 (저밀도)' if density < 5000 else '중간 (중밀도)' if density < 10000 else '장시간 소요 (고밀도)')}

재난 대응 시 고려사항:
{('- 고밀도 지역으로 신속한 대피가 어려울 수 있음' if density > 10000 else '')}
{('- 대규모 인구로 인해 대량의 대피 및 구호 자원 필요' if population > 100000 else '')}
{('- 고령 인구 비율이 높아 특별 지원 필요' if elderly_index > 100 else '')}
"""

            # append the summary/content to the output buffer instead of printing
            out_lines.append("="*60)
            out_lines.append(f"Matched area: {row.get('area_name', '')} ({row.get('administrative_level', '')}), code: {row.get('area_code', '')}")
            out_lines.append(f"인구: {int(population):,}명 | 밀도: {density:.1f}명/km² | 노령화지수: {elderly_index:.1f}")
            out_lines.append("문서 콘텐츠 (요약):")
            out_lines.append(content.strip())
            out_lines.append("="*60)

            documents.append(Document(
                page_content=content.strip(),
                metadata={
                    "source": "지역별_인구_경계_데이터",
                    "area_code": str(row.get('area_code', '')),
                    "area_name": str(row.get('area_name', '')),
                    "administrative_level": str(row.get('administrative_level', '')),
                    "population": int(population),
                    "density": float(density),
                    "type": "geospatial"
                }
            ))
        
        out_lines.append(f"Loaded {len(documents)} geospatial documents (filtered by query)")

        # write the collected output to a timestamped file for later review
        out_dir = Path("outputs") / "geospatial_queries"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"\s+", "_", f"{location_si}_{location_gu}_{location_dong}_{disaster}")
        out_path = out_dir / f"{ts}_{safe_name}.txt"
        out_path.write_text("\n".join(out_lines), encoding="utf-8")
    
    except Exception as e:
        print(f"Error loading geospatial data: {e}")
        import traceback
        traceback.print_exc()
    
    return documents


def build_vectorstores():
    """문서 분할 + 벡터DB 생성 (지리공간 데이터 포함)

    Adds support for separate law subsets (침수 / 정전) if those folders exist.
    Returns: vectordb_law, vectordb_flooding_law, vectordb_blackout_law, vectordb_manual, vectordb_basic, vectordb_past
    """
    law_docs = load_all_documents_to_list("Dataset/관련법령")
    # try to load specific law subfolders if present
    law_flooding_docs = []
    law_blackout_docs = []
    flooding_dir = "Dataset/관련법령/법령_풍수해"
    blackout_dir = "Dataset/관련법령/법령_정전"
    if os.path.exists(flooding_dir):
        law_flooding_docs = load_all_documents_to_list(flooding_dir)
    if os.path.exists(blackout_dir):
        law_blackout_docs = load_all_documents_to_list(blackout_dir)

    manual_docs = load_all_documents_to_list("Dataset/매뉴얼")
    basic_docs = load_all_documents_to_list("Dataset/기본데이터")
    past_docs = load_all_documents_to_list("Dataset/과거재난데이터")

    # Load geospatial documents and append to basic data (unchanged behavior)
    print("Loading geospatial documents...")
    geo_docs = load_geospatial_documents()
    if geo_docs:
        basic_docs.extend(geo_docs)
        print(f"Added {len(geo_docs)} geospatial documents to basic data")

    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)

    law_splits = splitter.split_documents(law_docs)
    law_flooding_splits = splitter.split_documents(law_flooding_docs) if law_flooding_docs else []
    law_blackout_splits = splitter.split_documents(law_blackout_docs) if law_blackout_docs else []
    manual_splits = splitter.split_documents(manual_docs)
    basic_splits = splitter.split_documents(basic_docs)
    past_splits = splitter.split_documents(past_docs)

    embeddings = load_embeddings()

    vectordb_law = FAISS.from_documents(law_splits, embeddings) if law_splits else None
    vectordb_flooding_law = FAISS.from_documents(law_flooding_splits, embeddings) if law_flooding_splits else None
    vectordb_blackout_law = FAISS.from_documents(law_blackout_splits, embeddings) if law_blackout_splits else None
    vectordb_manual = FAISS.from_documents(manual_splits, embeddings)
    vectordb_basic = FAISS.from_documents(basic_splits, embeddings)
    vectordb_past = FAISS.from_documents(past_splits, embeddings)

    return vectordb_law, vectordb_flooding_law, vectordb_blackout_law, vectordb_manual, vectordb_basic, vectordb_past