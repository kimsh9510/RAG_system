#벡터 DB구축 - > 지식그래프 DB로 변경 예정
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models import load_embeddings
import glob
import docx # python-docx
import os
import zipfile #hwpx 
import xml.etree.ElementTree as ET #hwxp
import pandas as pd

def load_all_documents_to_list(directory_path):
    all_documents = glob.glob(os.path.join(directory_path,"*"))
    
    documents=[]

    for file_path in all_documents:
        try:
            if file_path.endswith(".pdf"): #확장자 pdf로 끝나는 경우 
                reader=PdfReader(file_path)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    documents.append(Document(page_content=text, metadata={"source": file_path, "page": page_num}))

            elif file_path.endswith(".docx"): #docx로 끝나는 경우
                doc = docx.Document(file_path)
                full_text = "\n".join([para.text for para in doc.paragraphs])
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
                documents.append(Document(page_content=full_text, metadata={"source":file_path}))

            elif file_path.endswith(".txt"): #txt 
                with open(file_path,'r', encoding='utf-8') as f:
                    full_text=f.read()
                documents.append(Document(page_content=full_text, metadata={"source":file_path}))

            elif file_path.endswith(".xlsx"): #excel-모바일상황실 접수 현황 엑셀 기준
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

    return documents


def build_vectorstores():
    """문서 분할 + 벡터DB 생성"""
    law_docs = load_all_documents_to_list("Dataset/관련법령")
    manual_docs = load_all_documents_to_list("Dataset/매뉴얼")
    basic_docs = load_all_documents_to_list("Dataset/기본데이터")
    past_docs = load_all_documents_to_list("Dataset/과거재난데이터")

    #law_docs, manual_docs = load_all_documents_to_list("Dataset_for_test/과거재난데이터"), load_all_documents_to_list("Dataset_for_test/매뉴얼")
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)

    law_splits = splitter.split_documents(law_docs)
    manual_splits = splitter.split_documents(manual_docs)
    basic_splits = splitter.split_documents(basic_docs)
    past_splits = splitter.split_documents(past_docs)

    embeddings = load_embeddings()

    vectordb_law = FAISS.from_documents(law_splits, embeddings)
    vectordb_manual = FAISS.from_documents(manual_splits, embeddings)
    vectordb_basic = FAISS.from_documents(basic_splits, embeddings)
    vectordb_past = FAISS.from_documents(past_splits, embeddings)

    return vectordb_law, vectordb_manual, vectordb_basic, vectordb_past

