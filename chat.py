# chat.py
"""
Standalone chatbot with:
- GPT: full RAG context
- Llama3: short structured RAG context (like llm_node)
- Full chat history for both
- Extra cleaning & controls ONLY for Llama3
"""

import re
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

# ============================================================
# Silence transformers warnings (Llama3 etc.)
# ============================================================
try:
    # Silence transformers logging like:
    # "Starting from v4.46, the `logits` model output ..."
    from transformers.utils import logging as hf_logging  # type: ignore
    hf_logging.set_verbosity_error()
except Exception:
    pass

# Silence the annoying do_sample/temperature/top_p UserWarnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="transformers.generation.configuration_utils",
)


# ============================================================
# Out-of-scope filter
# ============================================================
def is_out_of_scope(msg: str) -> bool:
    out = [
        "연애", "사랑", "요리", "게임", "주식", "코딩", "알고리즘",
        "영화", "노래", "드라마", "메이크업", "건강", "운동",
        "심리", "보험", "취업", "여행", "맛집",
    ]
    dis = ["재난", "안전", "대응", "위험", "침수", "지진", "화재", "태풍", "정전", "대피", "피해", "수해"]

    return (any(k in msg for k in out) and not any(k in msg for k in dis)) or len(msg.strip()) < 2


# ============================================================
# Build FULL RAG context for GPT
# ============================================================
def build_full_context() -> str:
    (
        v_law,
        v_flood,
        v_blackout,
        v_manual,
        v_basic,
        v_pop,
        v_past,
    ) = build_vectorstores()

    state = {
        "query": "침수 발생 시 파생될 수 있는 재난 유형과 대응 매뉴얼",
        "location_si": "서울특별시",
        "location_gu": "서초구",
        "location_dong": "방배4동",
        "disaster": "침수",
    }

    parts: list[str] = []
    parts.append("[법]\n" + retrieval_law_node(v_law)(state)["law_ctx"])
    if v_flood:
        parts.append("[법_침수]\n" + retrieval_flooding_law_node(v_flood)(state)["law_flooding_ctx"])
    if v_blackout:
        parts.append("[법_정전]\n" + retrieval_blackout_law_node(v_blackout)(state)["law_blackout_ctx"])

    parts.append("[매뉴얼]\n" + retrieval_manual_node(v_manual)(state)["manual_ctx"])
    parts.append("[기본데이터]\n" + retrieval_basic_node(v_basic)(state)["basic_ctx"])
    parts.append("[GIS_인구]\n" + retrieval_population_node(v_pop)(state)["population_ctx"])
    parts.append("[과거재난데이터]\n" + retrieval_past_node(v_past)(state)["past_ctx"])

    return "\n\n".join(parts)


# ============================================================
# Build SHORT structured context for Llama3 (like llm_node)
# ============================================================
def build_llama3_context() -> str:
    """
    Same structure as llm_node, but carefully trimmed
    so the prompt is not gigantic.
    """
    (
        v_law,
        v_flood,
        v_blackout,
        v_manual,
        v_basic,
        v_pop,
        v_past,
    ) = build_vectorstores()

    state = {
        "query": "침수 발생 시 파생될 수 있는 재난 유형과 대응 매뉴얼",
        "location_si": "서울특별시",
        "location_gu": "서초구",
        "location_dong": "방배4동",
        "disaster": "침수",
    }

    parts: list[str] = []

    law = retrieval_law_node(v_law)(state)["law_ctx"]
    if len(law) > 1200:
        law = law[:1200]
    parts.append("[법]\n" + law)

    if v_flood:
        flood = retrieval_flooding_law_node(v_flood)(state)["law_flooding_ctx"]
        if len(flood) > 1200:
            flood = flood[:1200]
        parts.append("[법_침수]\n" + flood)

    manual = retrieval_manual_node(v_manual)(state)["manual_ctx"]
    if len(manual) > 1200:
        manual = manual[:1200]
    parts.append("[매뉴얼]\n" + manual)

    basic = retrieval_basic_node(v_basic)(state)["basic_ctx"]
    parts.append("[기본데이터]\n" + basic)

    pop = retrieval_population_node(v_pop)(state)["population_ctx"]
    if len(pop) > 800:
        pop = pop[:800]
    parts.append("[GIS_인구]\n" + pop)

    past = retrieval_past_node(v_past)(state)["past_ctx"]
    if len(past) > 800:
        past = past[:800]
    parts.append("[과거재난데이터]\n" + past)

    return "\n\n".join(parts)


# ============================================================
# Llama3 answer post-processing / cleaning
# ============================================================
def clean_llama3_answer(raw: str, user_question: str) -> str:
    """
    Cleans Llama3 output to:
    1) Remove prompt echo (법/매뉴얼/GIS context, 사용자 질문, etc.)
    2) Strip repeated disclaimers like:
       '위의 답변은 재난안전상담 AI가 제공하는 답변입니다...'
    3) Enforce short-ish output & keep only answer + 1-line suggestion.
    """
    if not raw:
        return ""

    text = raw

    # 1) If the model repeats "AI 답변:" style, keep only part after it
    if "AI 답변:" in text:
        text = text.split("AI 답변:", 1)[1].strip()

    # 2) If it followed the [답변] / [다음질문_1줄] format, parse it
    #    and drop anything after those sections.
    if "[답변]" in text:
        # Keep only from first [답변]
        text = text.split("[답변]", 1)[1]
        answer_part = text
        followup_part = ""

        if "[다음질문_1줄]" in text:
            answer_part, rest = text.split("[다음질문_1줄]", 1)
            # first non-empty line as follow-up question
            for line in rest.strip().splitlines():
                if line.strip():
                    followup_part = line.strip()
                    break

        # Rebuild in a clean, minimal format
        rebuilt = "[답변]\n" + answer_part.strip()
        if followup_part:
            rebuilt += "\n\n[다음질문_1줄]\n" + followup_part
        text = rebuilt.strip()

    # 3) Remove the repeated disclaimer lines completely
    disclaimer_pattern = re.compile(
        r"^위의 답변은 재난안전상담 AI가 제공하는 답변입니다.*$"
    )

    cleaned_lines = []
    for line in text.splitlines():
        if disclaimer_pattern.match(line.strip()):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines).strip()

    # 4) If the model still echoed the user question + 설명해줘 etc,
    #    that's fine; but we cut off any gigantic tail (safety).
    max_chars = 1500  # hard cap for runaway generations
    if len(text) > max_chars:
        # cut on a line boundary
        truncated = text[:max_chars]
        if "\n" in truncated:
            truncated = truncated.rsplit("\n", 1)[0]
        text = truncated.strip()

    return text.strip()


# ============================================================
# GPT chat (no special cleaning)
# ============================================================
def run_chat_gpt():
    print("[GPT-5-mini 로딩 중...]")
    llm = load_paid_gpt(model_id="gpt-5-mini")

    context = build_full_context()
    history: list[dict] = []

    system = (
        "당신은 재난안전대책본부의 친절한 AI 상담원입니다. "
        "항상 1000자 이내로, 핵심만 간단하게 답변하세요."
    )

    while True:
        user = input("\nYou: ").strip()
        if user.lower() in ["quit", "종료"]:
            break
        if is_out_of_scope(user):
            print("Assistant: 재난 관련 질문만 답변 가능합니다.")
            continue

        parts: list[str] = [system, "[상황정보]\n" + context]
        for h in history:
            parts.append(f"[{h['role']}]\n{h['text']}")
        parts.append(f"[user]\n{user}")

        prompt = "\n\n".join(parts)
        ans = llm.invoke(prompt)

        print("\nAssistant:\n", ans)
        history.append({"role": "user", "text": user})
        history.append({"role": "assistant", "text": ans})


# ============================================================
# Llama3 chat (short context, full history, cleaned output)
# ============================================================
def run_chat_llama3():
    print("[Llama3.1-8B 로딩 중...]")
    llm = load_llama3()

    context = build_llama3_context()
    history: list[dict] = []

    # NOTE: we force a strict output format here to make cleaning easy
    system = (
        "당신은 재난안전 상담을 도와주는 AI입니다.\n"
        "반드시 아래 형식을 지켜서만 답변하세요.\n\n"
        "[답변]\n"
        "- 사용자 질문에 대한 핵심 내용만 3~6문장으로 정리해서 작성\n"
        "- 불필요한 서론, 경고문, 반복 문장, 자기 소개는 절대 쓰지 말 것\n\n"
        "[다음질문_1줄]\n"
        "- 사용자가 이어서 물어볼 수 있는 한 문장짜리 질문 1개만 작성\n"
        "- 없으면 '다음 질문 예시: (예시 한 문장)' 형식으로 작성\n"
        "위 형식 이외의 텍스트는 절대 추가하지 마세요."
    )

    while True:
        user = input("\nYou: ").strip()
        if user.lower() in ["quit", "종료"]:
            break
        if is_out_of_scope(user):
            print("Assistant: 재난 관련 질문만 가능합니다.")
            continue

        # Structured prompt: system + 상황 + 모든 이전 대화 + 새 질문
        parts: list[str] = [system]
        parts.append("[핵심상황]\n" + context)

        for h in history:
            parts.append(f"이전대화-{h['role']}:\n{h['text']}")

        parts.append("사용자 질문:\n" + user)

        prompt = "\n\n".join(parts)
        raw_ans = llm.invoke(prompt)
        ans = clean_llama3_answer(raw_ans, user)

        print("\nAssistant:\n", ans)

        history.append({"role": "user", "text": user})
        history.append({"role": "assistant", "text": ans})


# ============================================================
# Main chooser
# ============================================================
if __name__ == "__main__":
    print("======= 모델 선택 =======")
    print("1) Llama3 (local, cleaned)")
    print("2) GPT-5-mini (API, full context)")
    c = input("모델 번호 입력: ").strip()

    if c == "1":
        run_chat_llama3()
    else:
        run_chat_gpt()
