#Langgraph 노드 구축
from typing import TypedDict
from operator import add
from langchain.schema import Document

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
    gis_ctx: str
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

def retrieval_past_node(vectordb_past):
    def node(state: State):
        q = state["query"]
        docs = vectordb_past.similarity_search(q, k=3)
        return {"past_ctx": "\n".join(d.page_content for d in docs)}
    return node

def retrieval_gis_node(vectordb_gis):
    def node(state: State):
        # Query the population vectorstore in the same way other retrieval nodes do.
        q = state.get("query") or ""
        docs = vectordb_gis.similarity_search(q, k=3)
        # Return the same field name used elsewhere so llm_node picks it up.
        return {"gis_ctx": "\n".join(d.page_content for d in docs)}
    return node

def llm_node(llm):
    def node(state: State):
        location_si = state.get("location_si") or "서울특별시"
        location_gu = state.get("location_gu") or "서초구"
        location_dong = state.get("location_dong") or "서초3동"
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
        if "gis_ctx" in state:
            parts.append("[GIS데이터]\n" + state["gis_ctx"])
        if "past_ctx" in state:
            parts.append("[과거재난데이터]\n" + state["past_ctx"])
        context = "\n\n".join(parts)

        ##prompt 수정 필요
        prompt = f"""당신은 지역재난안전대책본부의 통제관입니다.
                {location_si} {location_gu} {location_dong}에서 발생한 {disaster} 관련하여 재난 예측 및 대응 시나리오를 생성하려고 합니다.

                아래 문서는 법, 매뉴얼, 기본데이터, 과거재난 데이터를 통합하고 있습니다.
                {context}
                
                    대량의 물에 잠긴 지역에서 발생할 수 있는 전기 충격 및 화재 위험, 물중독 및 질병 위험, 구조물 파괴 및 붕괴 위험, 화재 및 폭발 위험에 대해
                    재난관리 역할 임무별(본부장, 차장, 통제관, 담당관, 현장대응담당자)로 발생할 수 있는 
                    재난관리 또는 권한 상의 문제점을 관련법령을 기반으로 검토해줘.
                    제공된 문서를 참고하되, 만약 문맥에서 잠재적이거나 일반적인 문제점이 추론된다면 당신의 일반 지식과 추론 능력을 활용하여 함께 설명해줘.
                
                """
        
        #{state.get("law_flooding_ctx", "")}
        answer = llm.invoke(prompt)
        return {"answer": answer}
    return node

def response_node(state: State):
    print("최종 답변:\n", state["answer"])
    return {}