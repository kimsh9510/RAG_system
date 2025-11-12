#Langgraph 노드 구축
from typing import TypedDict
from operator import add
from langchain.schema import Document
from pathlib import Path
import subprocess
import sys

class State(TypedDict, total=False):
    query: str
    location_si : str
    location_gu : str
    location_dong : str
    disaster : str
    law_ctx: str
    law_flooding_ctx : str
    law_blackout_ctx : str
    manual_ctx: str
    basic_ctx: str
    past_ctx: str
    population_ctx: str
    answer: str

def retrieval_law_node(vectordb_law):
    def node(state: State):
        q = state["query"]
        docs = vectordb_law.similarity_search(q, k=3)
        return {"law_ctx": "\n".join(d.page_content for d in docs)}
    return node

def retrieval_flooding_law_node(vectordb_flooding_law):
    def node(state: State):
        q = state["query"]
        docs = vectordb_flooding_law.similarity_search(q, k=3)
        return {"law_flooding_ctx": "\n".join(d.page_content for d in docs)}
    return node

def retrieval_blackout_law_node(vectordb_blackout_law):
    def node(state: State):
        q = state["query"]
        docs = vectordb_blackout_law.similarity_search(q, k=3)
        return {"law_blackout_ctx": "\n".join(d.page_content for d in docs)}
    return node

def retrieval_manual_node(vectordb_manual):
    def node(state: State):
        q = state["query"]
        docs = vectordb_manual.similarity_search(q, k=30)
        return {"manual_ctx": "\n".join(d.page_content for d in docs)}
    return node

def retrieval_basic_node(vectordb_basic):
    def node(state: State):
        q = state["query"]
        docs = vectordb_basic.similarity_search(q, k=3)
        return {"basic_ctx": "\n".join(d.page_content for d in docs)}
    return node

def retrieval_population_node(vectordb_population):
    def node(state: State):
        # Query the population vectorstore in the same way other retrieval nodes do.
        q = state.get("query") or ""
        docs = vectordb_population.similarity_search(q, k=3)
        # Return the same field name used elsewhere so llm_node picks it up.
        return {"population_ctx": "\n".join(d.page_content for d in docs)}
    return node

def retrieval_past_node(vectordb_past):
    def node(state: State):
        q = state["query"]
        docs = vectordb_past.similarity_search(q, k=3)
        return {"past_ctx": "\n".join(d.page_content for d in docs)}
    return node

def llm_node(llm):
    def node(state: State):
        location_si = state.get("location_si") or "서울시"
        location_gu = state.get("location_gu") or "서초구"
        location_dong = state.get("location_dong") or "서초동"
        disaster = state.get("disaster") or "재난"

        parts = []
        if "law_ctx" in state:
            parts.append("[법]\n" + state["law_ctx"])
        if "law_flooding_ctx" in state:
            parts.append("[법_침수]\n" + state["law_flooding_ctx"])
        if "law_blackout_ctx" in state:
            parts.append("[법_정전]\n" + state["law_blackout_ctx"])
        if "manual_ctx" in state:
            parts.append("[매뉴얼]\n" + state["manual_ctx"])
        if "basic_ctx" in state:
            parts.append("[기본데이터]\n" + state["basic_ctx"])
        if "population_ctx" in state:
            parts.append("[GIS_인구]\n" + state["population_ctx"])
        if "past_ctx" in state:
            parts.append("[과거재난데이터]\n" + state["past_ctx"])
        context = "\n\n".join(parts)

        ##prompt 수정 필요
        prompt = f"""당신은 지역재난안전대책본부의 통제관입니다.
                {location_si} {location_gu} {location_dong}에서 발생한 {disaster} 관련하여 재난 예측 및 대응 시나리오를 생성하려고 합니다.

                아래 문서는 법, 매뉴얼, 기본데이터, 과거재난 데이터를 통합하고 있습니다.
                {context}

                문서를 바탕으로 다음 두가지를 작성하세요.
                1. [연계 재난 탐지]
                "태풍"이 발생했을 때, 함께 발생하거나 영향을 줄 수 있는 연계 재난을 3가지 정도 나열하세요.
                각 재난은 왜 발생하는지(원인)와 어떤 피해로 이어지는지도 간단히 설명하세요.  
                
                2. [대응 시나리오]
                위에서 탐지된 각 연계 재난 유형별로, 단계별 대응 절차를 [법_{disaster}] 법령을 참고하여 제시하세요.
                
                """
        
        #{state.get("law_flooding_ctx", "")}
        answer = llm.invoke(prompt)
        return {"answer": answer}
    return node

def response_node(state: State):
    print("최종 답변:\n", state["answer"])
    return {}