#Langgraph 노드 구축
from typing import TypedDict
from operator import add
from langchain.schema import Document


class State(TypedDict, total=False):
    query: str
    # location fields (added)
    location_si : str
    location_gu : str
    location_dong : str
    disaster : str
    # contexts
    law_ctx: str
    law_flooding_ctx : str
    law_blackout_ctx : str
    manual_ctx: str
    basic_ctx: str
    past_ctx: str
    answer: str


def retrieval_law_node(vectordb_law):
    def node(state: State):
        q = state.get("query", "")
        docs = vectordb_law.similarity_search(q, k=3) if vectordb_law is not None else []
        return {"law_ctx": "\n".join(d.page_content for d in docs)}
    return node


def retrieval_flooding_law_node(vectordb_flooding_law):
    def node(state: State):
        q = state.get("query", "")
        docs = vectordb_flooding_law.similarity_search(q, k=3) if vectordb_flooding_law is not None else []
        return {"law_flooding_ctx": "\n".join(d.page_content for d in docs)}
    return node


def retrieval_blackout_law_node(vectordb_blackout_law):
    def node(state: State):
        q = state.get("query", "")
        docs = vectordb_blackout_law.similarity_search(q, k=3) if vectordb_blackout_law is not None else []
        return {"law_blackout_ctx": "\n".join(d.page_content for d in docs)}
    return node


def retrieval_manual_node(vectordb_manual):
    def node(state: State):
        q = state.get("query", "")
        # use larger k to favour manual coverage (as in updated repo)
        docs = vectordb_manual.similarity_search(q, k=30)
        return {"manual_ctx": "\n".join(d.page_content for d in docs)}
    return node


def retrieval_basic_node(vectordb_basic):
    def node(state: State):
        q = state.get("query", "")
        docs = vectordb_basic.similarity_search(q, k=3)

        # Separate geospatial and non-geospatial documents (preserve your geospatial integration)
        geo_docs = [d for d in docs if d.metadata.get("type") == "geospatial"]
        other_docs = [d for d in docs if d.metadata.get("type") != "geospatial"]

        # If we have geospatial docs, keep them; otherwise try a location-specific query
        if geo_docs:
            all_docs = docs
        else:
            # Try to get geospatial data by searching for location-related terms
            location_si = state.get("location_si", "")
            location_gu = state.get("location_gu", "")
            location_dong = state.get("location_dong", "")
            location_parts = [p for p in (location_si, location_gu, location_dong) if p]
            location_query = q + (" " + " ".join(location_parts) if location_parts else "") + " 지역 인구 밀도"
            location_docs = vectordb_basic.similarity_search(location_query, k=5)
            geo_docs = [d for d in location_docs if d.metadata.get("type") == "geospatial"][:2]
            all_docs = other_docs + geo_docs

        return {"basic_ctx": "\n".join(d.page_content for d in all_docs)}
    return node


def retrieval_past_node(vectordb_past):
    def node(state: State):
        q = state.get("query", "")
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
        if "past_ctx" in state:
            parts.append("[과거재난데이터]\n" + state["past_ctx"])
        context = "\n\n".join(parts)

        prompt = f"""당신은 지역재난안전대책본부의 통제관입니다.
                {location_si} {location_gu} {location_dong}에서 발생한 {disaster} 관련하여 재난 예측 및 대응 시나리오를 생성하려고 합니다.

                아래 문서는 법, 매뉴얼, 기본데이터, 과거재난 데이터를 통합하고 있습니다.
                {context}

                문서를 바탕으로 다음 두가지를 작성하세요. **지역 특성(인구밀도, 인구수, 고령화 지수 등)을 반드시 고려하세요.**

                1. [연계 재난 탐지]
                "{disaster}"이 발생했을 때, 함께 발생하거나 영향을 줄 수 있는 연계 재난을 3가지 정도 나열하세요.
                각 재난은 왜 발생하는지(원인)와 어떤 피해로 이어지는지도 간단히 설명하세요.
                **해당 지역({location_si} {location_gu} {location_dong})의 인구 밀도와 면적을 고려하여 예상 피해 규모를 추정하세요.**

                2. [대응 시나리오]
                위에서 탐지된 각 연계 재난 유형별로, 단계별 대응 절차를 [법_{disaster}] 법령 및 제공된 매뉴얼 문서를 참고하여 제시하세요.
                **해당 지역의 인구수와 특성(고령화 등)을 고려하여 필요한 대응 자원 규모와 우선순위를 명시하세요.**

                사용자 요청:
                {state.get('query', '')}
                """
        answer = llm.invoke(prompt)
        return {"answer": answer}
    return node


def response_node(state: State):
    print("최종 답변:\n", state["answer"]) if "answer" in state else print("최종 답변 없음")
    return {}