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

##4가지 data(기본데이터, 과거재난 데이터, 재난 관련 법령, 매뉴얼)로 확장
##각 data 리스트에 여러개의 파일 추가

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

        except Exception as e:
            # 해당 파일은 건너뛰고 오류 메시지를 출력
            print(f"Error processing {file_path}: {e}")

    return documents


def build_vectorstores():
    """문서 분할 + 벡터DB 생성"""
    law_docs, manual_docs = load_all_documents_to_list("Dataset/관련법령"), load_all_documents_to_list("Dataset/매뉴얼")
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)

    law_splits = splitter.split_documents(law_docs)
    manual_splits = splitter.split_documents(manual_docs)

    embeddings = load_embeddings()
    vectordb_law = FAISS.from_documents(law_splits, embeddings)
    vectordb_manual = FAISS.from_documents(manual_splits, embeddings)

    return vectordb_law, vectordb_manual

