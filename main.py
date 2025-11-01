import sys
import os
from langgraph.graph import StateGraph, START, END
from knowledge_base_copy1 import build_vectorstores
from models import load_llm, load_solar_pro, load_EXAONE, load_llama3, load_solar_pro2, load_EXAONE_api
from nodes import State, retrieval_law_node, retrieval_manual_node, retrieval_basic_node, retrieval_past_node, llm_node, response_node

#벡터 db와 LLM모델 로드
vectordb_law, vectordb_manual, vectordb_basic, vectordb_past = build_vectorstores()
# Use the appropriate LLM loading function from models.py
llm = load_llama3() # or load_solar_pro(), load_EXAONE(), load_llama3(), load_llm()

#langgraph 정의
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

if __name__ == "__main__":
    app = graph.compile()
    #query 내용을 기반으로 문서 탐색
    result = app.invoke({"query": "태풍 발생 시 파생될 수 있는 재난 유형과 대응 매뉴얼"})