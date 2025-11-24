import sys
import os

from langgraph.graph import StateGraph, START, END
from knowledge_base_copy1 import build_vectorstores
from models import load_llm, load_solar_pro, load_EXAONE, load_llama3, load_solar_pro2, load_EXAONE_api, load_gpt, load_paid_gpt
from nodes import (
    State,
    retrieval_law_node,
    retrieval_flooding_law_node,
    retrieval_blackout_law_node,
    retrieval_manual_node,
    retrieval_basic_node,
    retrieval_past_node,
    retrieval_population_node,
    llm_node,
    response_node,
)


# ============================================================
# ① Out-of-scope detection helper
# ============================================================
def is_out_of_scope(user_msg: str) -> bool:
    """Return True if the message is irrelevant to disaster/safety topics."""
    out_keywords = [
        "연애", "사랑", "요리", "게임", "주식", "코딩", "알고리즘",
        "영화", "노래", "드라마", "메이크업", "건강", "운동",
        "심리", "보험", "취업", "여행", "맛집"
    ]

    disaster_keywords = [
        "재난", "안전", "대응", "위험", "침수", "지진",
        "화재", "태풍", "정전", "대피", "피해", "수해"
    ]

    # If message contains irrelevant keywords AND no disaster keywords → out
    if any(k in user_msg for k in out_keywords) and not any(d in user_msg for d in disaster_keywords):
        return True

    # If message is too short (nonsense)
    if len(user_msg.strip()) < 2:
        return True

    return False


# ============================================================
# Load vector DBs and LLM
# ============================================================
(
    vectordb_law,
    vectordb_flooding_law,
    vectordb_blackout_law,
    vectordb_manual,
    vectordb_basic,
    vectordb_population,
    vectordb_past,
) = build_vectorstores()

llm = load_paid_gpt()  # or another model


def build_graph(disaster: str = None):
    graph = StateGraph(State)

    graph.add_node("retrieval_law", retrieval_law_node(vectordb_law))
    graph.add_node("retrieval_manual", retrieval_manual_node(vectordb_manual))
    graph.add_node("retrieval_basic", retrieval_basic_node(vectordb_basic))
    graph.add_node("retrieval_population", retrieval_population_node(vectordb_population))
    graph.add_node("retrieval_past", retrieval_past_node(vectordb_past))
    graph.add_node("llm", llm_node(llm))
    graph.add_node("response", response_node)

    graph.add_edge(START, "retrieval_law")
    graph.add_edge(START, "retrieval_manual")
    graph.add_edge(START, "retrieval_basic")
    graph.add_edge(START, "retrieval_population")
    graph.add_edge(START, "retrieval_past")
    graph.add_edge("retrieval_law", "llm")
    graph.add_edge("retrieval_manual", "llm")
    graph.add_edge("retrieval_basic", "llm")
    graph.add_edge("retrieval_population", "llm")
    graph.add_edge("retrieval_past", "llm")
    graph.add_edge("llm", "response")
    graph.add_edge("response", END)

    if disaster == "침수" and vectordb_flooding_law is not None:
        graph.add_node("retrieval_flooding_law", retrieval_flooding_law_node(vectordb_flooding_law))
        graph.add_edge(START, "retrieval_flooding_law")
        graph.add_edge("retrieval_flooding_law", "llm")
        print("침수 관련 노드 추가 완료")

    if disaster == "정전" and vectordb_blackout_law is not None:
        graph.add_node("retrieval_blackout_law", retrieval_blackout_law_node(vectordb_blackout_law))
        graph.add_edge(START, "retrieval_blackout_law")
        graph.add_edge("retrieval_blackout_law", "llm")
        print("정전 관련 노드 추가 완료")

    return graph.compile()


# ============================================================
# Main Chat
# ============================================================
if __name__ == "__main__":
    location_si = "서울특별시"
    location_gu = "서초구"
    location_dong = "방배4동"
    disaster = "침수"
    initial_query = f"{disaster} 발생 시 파생될 수 있는 재난 유형과 대응 매뉴얼"

    state = {
        "query": initial_query,
        "location_si": location_si,
        "location_gu": location_gu,
        "location_dong": location_dong,
        "disaster": disaster,
    }

    law_ctx = retrieval_law_node(vectordb_law)(state)["law_ctx"]
    manual_ctx = retrieval_manual_node(vectordb_manual)(state)["manual_ctx"]
    basic_ctx = retrieval_basic_node(vectordb_basic)(state)["basic_ctx"]
    population_ctx = retrieval_population_node(vectordb_population)(state)["population_ctx"]
    past_ctx = retrieval_past_node(vectordb_past)(state)["past_ctx"]
    law_flooding_ctx = ""
    law_blackout_ctx = ""

    if disaster == "침수" and vectordb_flooding_law:
        law_flooding_ctx = retrieval_flooding_law_node(vectordb_flooding_law)(state)["law_flooding_ctx"]

    if disaster == "정전" and vectordb_blackout_law:
        law_blackout_ctx = retrieval_blackout_law_node(vectordb_blackout_law)(state)["law_blackout_ctx"]

    context_parts = []
    if law_ctx:
        context_parts.append("[법]\n" + law_ctx)
    if law_flooding_ctx:
        context_parts.append("[법_침수]\n" + law_flooding_ctx)
    if law_blackout_ctx:
        context_parts.append("[법_정전]\n" + law_blackout_ctx)
    if manual_ctx:
        context_parts.append("[매뉴얼]\n" + manual_ctx)
    if basic_ctx:
        context_parts.append("[기본데이터]\n" + basic_ctx)
    if population_ctx:
        context_parts.append("[GIS_인구]\n" + population_ctx)
    if past_ctx:
        context_parts.append("[과거재난데이터]\n" + past_ctx)

    context = "\n\n".join(context_parts)

    system_prompt = (
        "당신은 재난안전대책본부의 친절한 AI 상담원입니다. "
        "질문에 답변하거나, 필요한 정보를 안내하세요."
    )

    print("\n[Chatbot 모드 시작 - Quit 또는 종료 입력시 종료]\n")

    chat_history = []

    try:
        while True:
            user = input("You: ").strip()

            # --- 1) Quit command ---
            if user.lower() in ["quit", "종료"]:
                print("Chat ended. 종료합니다.")
                break

            # --- 2) Out-of-scope detection ---
            if is_out_of_scope(user):
                print("\nAssistant:\n죄송합니다. 이 질문은 재난·안전 상담 범위를 벗어난 내용입니다.\n"
                      "재난, 안전, 대응 매뉴얼, 대피, GIS 인구 등과 관련된 질문을 해주세요.\n")
                continue

            # --- 3) Normal flow ---
            prompt_parts = [system_prompt, "[상황정보]\n" + context, ""]
            for turn in chat_history:
                prompt_parts.append(f"[{turn['role']}]\n{turn['text']}")
            prompt_parts.append("[user]\n" + user)

            final_prompt = "\n\n".join(prompt_parts)
            resp = llm.invoke(final_prompt)

            print("\nAssistant:\n", resp)

            chat_history.append({"role": "user", "text": user})
            chat_history.append({"role": "assistant", "text": resp})

    except KeyboardInterrupt:
        print("\n종료 (Ctrl-C)")
