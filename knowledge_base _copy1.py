#벡터 DB구축 - > 지식그래프 DB로 변경 예정
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models import load_embeddings

##4가지 data(기본데이터, 과거재난 데이터, 재난 관련 법령, 매뉴얼)로 확장
##각 data 리스트에 여러개의 파일 추가
def load_documents():
    """PDF 로드 & Document 리스트 반환"""
    law_data = PdfReader("Dataset/관련법.pdf")
    manual_data = PdfReader("Dataset/화재상황시나리오.pdf")
    #basic_data = PdfReader("Dataset/-.pdf")
    #past_data = PdfReader("Dataset/-.pdf")

    law_docs = [Document(page_content=p.extract_text(), metadata={"source": "관련법.pdf"}) 
                for p in law_data.pages]
    manual_docs = [Document(page_content=p.extract_text(), metadata={"source": "화재상황시나리오.pdf"}) 
                     for p in manual_data.pages]
    
    return law_docs, manual_docs


def build_vectorstores():
    """문서 분할 + 벡터DB 생성"""
    law_docs, manual_docs = load_documents()
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)

    law_splits = splitter.split_documents(law_docs)
    manual_splits = splitter.split_documents(manual_docs)

    embeddings = load_embeddings()
    vectordb_law = FAISS.from_documents(law_splits, embeddings)
    vectordb_manual = FAISS.from_documents(manual_splits, embeddings)

    return vectordb_law, vectordb_manual

