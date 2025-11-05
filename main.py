import sys
import os
from langgraph.graph import StateGraph, START, END
from knowledge_base_copy1 import build_vectorstores
from models import load_qwen, load_solar_pro, load_solar_pro2, load_llama3, load_EXAONE
from nodes import State, retrieval_law_node, retrieval_flooding_law_node, retrieval_blackout_law_node, retrieval_manual_node, retrieval_basic_node, retrieval_past_node, llm_node, response_node

#벡터 db와 LLM모델 로드
vectordb_law, vectordb_flooding_law, vectordb_blackout_law, vectordb_manual, vectordb_basic, vectordb_past = build_vectorstores()
llm = load_qwen()

#langgraph 정의
def build_graph(disaster: str):
    graph = StateGraph(State)
    graph.add_node("retrieval_law", retrieval_law_node(vectordb_law))
    graph.add_node("retrieval_manual", retrieval_manual_node(vectordb_manual))
    graph.add_node("retrieval_basic", retrieval_basic_node(vectordb_basic))
    graph.add_node("retrieval_past", retrieval_past_node(vectordb_past))
    graph.add_node("llm", llm_node(llm))
    graph.add_node("response", response_node)

    ##langgraph 엣지 추가
    graph.add_edge(START, "retrieval_law")
    graph.add_edge(START, "retrieval_manual")
    graph.add_edge(START, "retrieval_basic")
    graph.add_edge(START, "retrieval_past")
    graph.add_edge("retrieval_law", "llm")
    graph.add_edge("retrieval_manual", "llm")
    graph.add_edge("retrieval_basic", "llm")
    graph.add_edge("retrieval_past", "llm")
    graph.add_edge("llm", "response")
    graph.add_edge("response", END)

    #사용자가 입력한 재난(침수, 정전)에 따라 langgraph가 참조하는 법령 노드 추가
    if disaster == "침수":
        graph.add_node("retrieval_flooding_law", retrieval_flooding_law_node(vectordb_flooding_law))
        graph.add_edge(START, "retrieval_flooding_law")
        graph.add_edge("retrieval_flooding_law", "llm")
        print("침수 관련 노드 추가 완료")

    elif disaster == "정전":
        graph.add_node("retrieval_blackout_law", retrieval_blackout_law_node(vectordb_blackout_law))
        graph.add_edge(START, "retrieval_blackout_law")
        graph.add_edge("retrieval_blackout_law", "llm")
        print("정전 관련 노드 추가 완료")
    
    return graph.compile()

if __name__ == "__main__":
    #사용자 입력값(침수, 정전)
    disaster = "침수"
    location_si = "서울시"
    location_gu = "서초구"
    location_dong = "서초동"

    #langgraph 생성
    app = build_graph(disaster)
    
    #query 내용을 기반으로 문서 탐색
    result = app.invoke({
        "query": f"{disaster} 발생 시 파생될 수 있는 재난 유형과 대응 매뉴얼",
        "location_si": location_si,
        "location_gu": location_gu,
        "location_dong" : location_dong,
        "disaster": disaster
    })
    
    #실제 추출되는 문서내용 출력
    #docs = vectordb_manual.similarity_search("태풍 또는 풍수해 발생 시 파생 재난 유형, 연계재난, 대응 절차, 긴급복구 관련 법령", k=5)
    #for i, d in enumerate(docs, 1):
    #    print(f"\n[{i}] 파일: {d.metadata.get('source', 'unknown')}")
    #    print(d.page_content[:300]) 