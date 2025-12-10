from gemini_rag import gemini_rag_answer

if __name__ == "__main__":
    print("\n=== Gemini RAG Test ===\n")

    question = "주어진 동내에서 침수 시 어떤 연계 재난들이 함께 발생할 수 있나요?"
    answer = gemini_rag_answer(question)

    print("\n=== 최종 답변 ===\n")
    print(answer)
