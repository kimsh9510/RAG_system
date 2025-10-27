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

##retrieval_past_node 추가
def retrieval_law_node(vectordb_law):
    def node(state: State):
        q = state["query"]
        docs = vectordb_law.similarity_search(q, k=2)
        return {"law_ctx": "\n".join(d.page_content for d in docs)}
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
        docs = vectordb_basic.similarity_search(q, k=2)
        return {"basic_ctx": "\n".join(d.page_content for d in docs)}
    return node

def retrieval_past_node(vectordb_past):
    def node(state: State):
        q = state["query"]
        docs = vectordb_past.similarity_search(q, k=2)
        return {"past_ctx": "\n".join(d.page_content for d in docs)}
    return node

def llm_node(llm):
    def node(state: State):
        #디버깅 코드
        law_content = state.get("law_ctx", "")
        manual_content = state.get("manual_ctx", "")
        basic_content = state.get("basic_ctx", "")
        past_content = state.get("past_ctx", "")
        #---
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
                아래 문서는 법, 매뉴얼, 기본데이터, 과거재난 데이터를 통합하고 있습니다.
                {context}


{context}

질문: {state['query']}
답변:"""
        answer = llm.invoke(prompt)
        return {"answer": answer}
    return node

def response_node(state: State):
    print("최종 답변:\n", state["answer"])
    return {}