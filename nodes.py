#Langgraph 노드 구축
from typing import TypedDict
from operator import add
from langchain.schema import Document

class State(TypedDict, total=False):
    query: str
    law_ctx: str
    manual_ctx: str
    basic_ctx: str
    past_ctx: str
    answer: str

def retrieval_law_node(vectordb_law):
    def node(state: State):
        q = state["query"]
        docs = vectordb_law.similarity_search(q, k=2)
        return {"law_ctx": "\n".join(d.page_content for d in docs)}
    return node

def retrieval_manual_node(vectordb_manual):
    def node(state: State):
        q = state["query"]
        docs = vectordb_manual.similarity_search(q, k=2)
        return {"manual_ctx": "\n".join(d.page_content for d in docs)}
    return node

def retrieval_basic_node(vectordb_basic):
    def node(state: State):
        q = state["query"]
        docs = vectordb_basic.similarity_search(q, k=3)
        
        # Separate geospatial and non-geospatial documents
        geo_docs = [d for d in docs if d.metadata.get("type") == "geospatial"]
        other_docs = [d for d in docs if d.metadata.get("type") != "geospatial"]
        
        # If we have geospatial docs, fetch a few more to ensure good coverage
        if geo_docs:
            all_docs = docs
        else:
            # Try to get geospatial data by searching for location-related terms
            location_query = q + " 지역 인구 밀도"
            location_docs = vectordb_basic.similarity_search(location_query, k=5)
            geo_docs = [d for d in location_docs if d.metadata.get("type") == "geospatial"][:2]
            all_docs = other_docs + geo_docs
        
        return {"basic_ctx": "\n".join(d.page_content for d in all_docs)}
    return node

def retrieval_past_node(vectordb_past):
    def node(state: State):
        q = state["query"]
        docs = vectordb_past.similarity_search(q, k=2)
        return {"past_ctx": "\n".join(d.page_content for d in docs)}
    return node

def llm_node(llm):
    def node(state: State):
        parts = []
        if "law_ctx" in state:
            parts.append("[법]\n" + state["law_ctx"])
        if "manual_ctx" in state:
            parts.append("[매뉴얼]\n" + state["manual_ctx"])
        if "basic_ctx" in state:
            parts.append("[기본데이터]\n" + state["basic_ctx"])
        if "past_ctx" in state:
            parts.append("[과거재난데이터]\n" + state["past_ctx"])
        context = "\n\n".join(parts)

        ##사용자 요청: 쿼리 문이 꼭 추가되어야 하는지 확인
        prompt = f"""당신은 지역재난안전대책본부의 통제관입니다.
                아래 문서는 법, 매뉴얼, 기본데이터, 과거재난 데이터, 그리고 지역별 인구 및 경계 정보를 통합하고 있습니다.
                {context}

                문서를 바탕으로 다음 두가지를 작성하세요. **지역 특성(인구밀도, 인구수, 고령화 지수 등)을 반드시 고려하세요.**
                
                1. [연계 재난 탐지]
                "태풍"이 발생했을 때, 함께 발생하거나 영향을 줄 수 있는 연계 재난을 3가지 정도 나열하세요.
                각 재난은 왜 발생하는지(원인)와 어떤 피해로 이어지는지도 간단히 설명하세요.
                **해당 지역의 인구 밀도와 면적을 고려하여 예상 피해 규모를 추정하세요.**
                
                2. [대응 시나리오]
                위에서 탐지된 각 연계 재난 유형별로, 아래 매뉴얼 문서를 바탕으로 단계별 대응 절차를 제시하세요.
                **해당 지역의 인구수와 특성(고령화 등)을 고려하여 필요한 대응 자원 규모와 우선순위를 명시하세요.**
                {state.get("manual_ctx", "")}

                사용자 요청:
                {state['query']} 
                """
        answer = llm.invoke(prompt)
        return {"answer": answer}
    return node

def response_node(state: State):
    print("최종 답변:\n", state["answer"])
    return {}