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
    
    selected_model: str
    
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

def llm_node(models_map):
    def node(state: State):
        user_query = state.get("query", "요청 내용 없음")
        location_si = state.get("location_si") or "서울시"
        location_gu = state.get("location_gu") or "서초구"
        
        #모델 선택 : 값이 없거나 이상하면 llama로
        location_dong = state.get("location_dong") or "서초동"
        disaster = state.get("disaster") or "재난"
        
        target_model_key = state.get("selected_model", "llama3")
        current_llm = models_map.get(target_model_key)
        if not current_llm:
            print(f" 경고: '{target_model_key}' 모델을 찾을 수 없습니다. 기본 모델(llama)을 사용합니다.")
            current_llm = models_map.get("llama3")
        
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
    
        ##prompt 수정 필요
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
                {user_query} 에 대한 답변을 알려주세요.
                
                [참조 문서]
                {context}
                
                대답:
                
                """
        answer = current_llm.invoke(prompt)
        
        #  응답이 객체(AIMessage)인 경우 content만 추출, 문자열이면 그대로 사용
        raw_content = answer.content if hasattr(answer, 'content') else str(answer)

        #  "대답:" 키워드 기준으로 자르기
        separator = "대답:"
        if separator in raw_content:
            final_answer = raw_content.split(separator, 1)[1].strip()
        else:
            final_answer = raw_content.strip()
        return {"answer": final_answer}
    return node

def response_node(state: State):
    print("최종 답변:\n", state["answer"])
    return {}
