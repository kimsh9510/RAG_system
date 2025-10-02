import sys
import os
print("--- Python Debugging Info ---")
print(f"실행 중인 파이썬 경로 (sys.executable): {sys.executable}")
print("\n모듈을 찾는 경로 목록 (sys.path):")
for p in sys.path:
    print(f"  - {p}")
print("--- End Debugging Info ---\n")


from langgraph.graph import StateGraph, START, END
from knowledge_base_copy1 import build_vectorstores
from models import load_llm
from nodes import State, retrieval_law_node, retrieval_manual_node, llm_node, response_node

#벡터 db와 LLM모델 로드
vectordb_law, vectordb_manual = build_vectorstores()
llm = load_llm()

#langgraph 정의
graph = StateGraph(State)
graph.add_node("retrieval_law", retrieval_law_node(vectordb_law))
graph.add_node("retrieval_manual", retrieval_manual_node(vectordb_manual))
graph.add_node("llm", llm_node(llm))
graph.add_node("response", response_node)

##langgraph 노드 추가
graph.add_edge(START, "retrieval_law")
graph.add_edge(START, "retrieval_manual")
graph.add_edge("retrieval_law", "llm")
graph.add_edge("retrieval_manual", "llm")
graph.add_edge("llm", "response")
graph.add_edge("response", END)

if __name__ == "__main__":
    app = graph.compile()
    result = app.invoke({"query": "문서를 기반으로 본부장의 역할을 설명해줘"})