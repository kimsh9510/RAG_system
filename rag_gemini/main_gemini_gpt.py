from gemini_rag import gemini_retrieve_chunks
from models import load_paid_gpt

gpt = load_paid_gpt("gpt-5.1")

if __name__ == "__main__":
    question = "서초구 침수 시 어떤 연계재난이 함께 발생하나요?"

    # Step 1: Gemini retrieves chunks
    chunks = gemini_retrieve_chunks(question)

    # Step 2: GPT produces final answer
    prompt = f"""
사용자 질문:
{question}

검색된 문서:
{chunks}

위 문서를 기반으로 정확하고 간결하게 답변하세요.
"""

    answer = gpt.invoke(prompt)
    print("\n=== GPT Final Answer ===\n")
    print(answer)
