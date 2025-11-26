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

        MAX_BLOCK = 15000 # 각 문서의 max token 수, 총 150000 이하가 되어야함
        # 단순 슬라이싱. llm활용해서 요약 가능
        def trim(text, max_len=MAX_BLOCK):
            return text[:max_len] if len(text) > max_len else text

        parts = []
        if "law_ctx" in state:
            parts.append("[법]\n" + trim(state["law_ctx"]))
        if "law_flooding_ctx" in state:
            parts.append("[법_침수]\n" + trim(state["law_flooding_ctx"]))
        if "law_blackout_ctx" in state:
            parts.append("[법_정전]\n" + trim(state["law_blackout_ctx"]))
        if "manual_ctx" in state:
            parts.append("[매뉴얼]\n" + trim(state["manual_ctx"]))
        if "basic_ctx" in state:
            parts.append("[기본데이터]\n" + trim(state["basic_ctx"]))
        if "gis_ctx" in state:
            parts.append("[GIS데이터]\n" + trim(state["gis_ctx"]))
        if "past_ctx" in state:
            parts.append("[과거재난데이터]\n" + trim(state["past_ctx"]))
        context = "\n\n".join(parts)
        print("context length : ",len(context))

        prompt = f"""당신은 지역재난안전대책본부의 통제관입니다.
        
                [분석 대상]
                지역 : {location_si} {location_gu} {location_dong}
                재난 : {disaster}
                
                아래 제공된 참조 문서에는 다양한 유형의 정보가 포함되어 있으며, 각 문서는 꺾쇠([])를 통해 구분됩니다.
                - [법], [법_침수] : 관련 법령 및 행정 지침
                - [매뉴얼] : 재난 대응 및 조치 매뉴얼
                - [기본데이터] : 재난 관련 일반 데이터
                - [GIS데이터] : 인구·지형·시설 등 공간 기반 데이터
                - [과거재난데이터] : 과거 사례 및 상황 정보

                아래 참조 문서를 기반으로, 
                재난 대응 및 지원 과정에 발생할 수 있는 위협요인들을 설명해주세요.
                
                [참조 문서]
                {context}
                
                대답:
                
                """
        #[GIS데이터]는 분석의 핵심 자료이므로, 답변을 작성할 때 GIS 정보에 기반한 근거를 명시적으로 포함해줘.
        #답변을 작성할 때는 반드시 [GIS데이터]의 구체적인 수치와 지표를 근거로 삼아 이유와 함께 설명해줘
        answer = llm.invoke(prompt)
        
        # GPU 메모리 해제 - 누적캐시 없애기
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        return {"answer": answer}
    return node

def response_node(state: State):
    print("최종 답변:\n", state["answer"])
    return {}