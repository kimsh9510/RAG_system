# chat.py — Demonstration version (Llama3 uses EXACT original nodes.py RAG with NO trimming)

import warnings

from knowledge_base_copy1 import build_vectorstores
from models import load_llama3, load_paid_gpt
from nodes import (
    retrieval_law_node,
    retrieval_flooding_law_node,
    retrieval_blackout_law_node,
    retrieval_manual_node,
    retrieval_basic_node,
    retrieval_past_node,
    retrieval_population_node,
)


# Silence warning noise
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

warnings.filterwarnings("ignore")


# ============================================================
# EXACT SAME CONTEXT AS llm_node (NO trimming)
# ============================================================
def build_context_exact_nodes(query, disaster):
    """
    This builds the context EXACTLY like nodes.py -> llm_node
    NO trimming. NO slicing. NO summarization.
    This will break Llama3 because context is massive.
    """

    (
        vectordb_law,
        vectordb_flooding_law,
        vectordb_blackout_law,
        vectordb_manual,
        vectordb_basic,
        vectordb_population,
        vectordb_past,
    ) = build_vectorstores()

    state = {
        "query": query,
        "location_si": "서울특별시",
        "location_gu": "서초구",
        "location_dong": "방배4동",
        "disaster": disaster,
    }

    parts = []

    # EXACT behavior of nodes.py
    parts.append("[법]\n" + retrieval_law_node(vectordb_law)(state)["law_ctx"])

    if disaster == "침수" and vectordb_flooding_law:
        parts.append("[법_침수]\n" + retrieval_flooding_law_node(vectordb_flooding_law)(state)["law_flooding_ctx"])

    if disaster == "정전" and vectordb_blackout_law:
        parts.append("[법_정전]\n" + retrieval_blackout_law_node(vectordb_blackout_law)(state)["law_blackout_ctx"])

    parts.append("[매뉴얼]\n" + retrieval_manual_node(vectordb_manual)(state)["manual_ctx"])
    parts.append("[기본데이터]\n" + retrieval_basic_node(vectordb_basic)(state)["basic_ctx"])
    parts.append("[GIS_인구]\n" + retrieval_population_node(vectordb_population)(state)["population_ctx"])
    parts.append("[과거재난데이터]\n" + retrieval_past_node(vectordb_past)(state)["past_ctx"])

    # This can exceed 50k characters, especially manual_ctx with k=30
    context = "\n\n".join(parts)
    return context


# ============================================================
# EXACT SAME PROMPT FORMAT AS llm_node (NO trimming)
# ============================================================
# def build_prompt(context, disaster, loc_si, loc_gu, loc_dong):
#     return f"""
# 당신은 지역재난안전대책본부의 통제관입니다.
# {loc_si} {loc_gu} {loc_dong}에서 발생한 {disaster} 관련하여 재난 예측 및 대응 시나리오를 생성하려고 합니다.

# 아래 문서는 법, 매뉴얼, 기본데이터, 과거재난 데이터를 통합하고 있습니다.
# {context}

# 문서를 바탕으로 다음 두가지를 작성하세요.

# 1. [연계 재난 탐지]
# "{disaster}"이 발생했을 때, 함께 발생하거나 영향을 줄 수 있는 연계 재난을 3가지 정도 나열하세요.
# 각 재난은 왜 발생하는지(원인)와 어떤 피해로 이어지는지도 간단히 설명하세요.

# 2. [대응 시나리오]
# 위에서 탐지된 각 연계 재난 유형별로, 단계별 대응 절차를 [법_{disaster}] 법령을 참고하여 제시하세요.
# """

# ============================================================
# REVISED PROMPT FORMAT FOR CHAT
# ============================================================
def build_prompt(context, disaster, loc_si, loc_gu, loc_dong):
    return f"""
    당신은 재난안전대책본부의 친절한 AI 상담원입니다.
    {loc_si} {loc_gu} {loc_dong}에서 발생한 {disaster} 관련하여 사용자의 질문에 답변하려고 합니다.
    
    아래 문서는 법, 매뉴얼, 기본데이터, 과거재난 데이터를 통합하고 있습니다.
    {context}
    문서를 바탕으로 사용자의 질문에 답변하세요.
    항상 1000자 이내로, 핵심만 간단하게 답변하세요.
    """


# ============================================================
# Llama3 chat using EXACT nodes.py system (NO TRIMMING)
# ============================================================
def run_llama3_raw():
    print("[Llama3.1-8B 로딩 중...]")
    llm = load_llama3()

    # Build giant context like original LangGraph system
    context = build_context_exact_nodes(
        query="침수 발생 시 파생될 수 있는 재난 유형과 대응 매뉴얼",
        disaster="침수",
    )

    loc_si = "서울특별시"
    loc_gu = "서초구"
    loc_dong = "방배4동"
    disaster = "침수"

    # Always reuse the SAME full context (like llm_node)
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in ["quit", "종료"]:
            break

        # Build prompt EXACTLY same as llm_node
        prompt = build_prompt(context, disaster, loc_si, loc_gu, loc_dong)

        # Add user question at the end
        prompt += f"\n\n[사용자 질문]\n{user}\n"

        # This is where Llama3 breaks
        answer = llm.invoke(prompt)

        print("\nAssistant:\n", answer)


# ============================================================
# Paid GPT chat using EXACT nodes.py system
# ============================================================
def run_paid_gpt_raw():
    print("[Paid GPT 모델 로딩 중...]")
    llm = load_paid_gpt()

    # Build giant context like original LangGraph system
    context = build_context_exact_nodes(
        query="침수 발생 시 파생될 수 있는 재난 유형과 대응 매뉴얼",
        disaster="침수",
    )

    loc_si = "서울특별시"
    loc_gu = "서초구"
    loc_dong = "방배4동"
    disaster = "침수"

    # Always reuse the SAME full context (like llm_node)
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in ["quit", "종료"]:
            break

        # Build prompt EXACTLY same as llm_node
        prompt = build_prompt(context, disaster, loc_si, loc_gu, loc_dong)

        # Add user question at the end
        prompt += f"\n\n[사용자 질문]\n{user}\n"

        answer = llm.invoke(prompt)

        print("\nAssistant:\n", answer)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Choose which model to run
    print("Select model: [1] Llama3 [2] Paid GPT")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "2":
        run_paid_gpt_raw()
    else:
        run_llama3_raw()
