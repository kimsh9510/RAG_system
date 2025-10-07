import sys
import os
from langgraph.graph import StateGraph, START, END
from knowledge_base_copy1 import build_vectorstores
from models import load_solar_pro, load_EXAONE, load_llama3
from nodes import State, retrieval_law_node, retrieval_manual_node, retrieval_basic_node, retrieval_past_node, llm_node, response_node

#벡터 db와 LLM모델 로드
vectordb_law, vectordb_manual, vectordb_basic, vectordb_past = build_vectorstores()
# Use the appropriate LLM loading function from models.py
llm = load_llama3()  # or load_solar_pro(), load_EXAONE()

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
    result = app.invoke({"query": "문서를 기반으로 본부장의 역할을 설명해줘"})

    # Optional: quick EXAONE demo. Set environment variable USE_EXAONE=1 to run.
    if os.environ.get("USE_EXAONE") == "1":
        print("\nRunning EXAONE demo (this may load a large model and take time)...")
        model, tokenizer = load_EXAONE()

        # Non-reasoning example
        prompt = "Explain how wonderful you are"
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        out = model.generate(
            input_ids.to(model.device),
            max_new_tokens=128,
            do_sample=False,
            temperature=0.3,
        )
        print("\nEXAONE non-reasoning output:\n", tokenizer.decode(out[0]))

        # Reasoning example (enable thinking)
        messages = [{"role": "user", "content": "Which one is bigger, 3.12 vs 3.9?"}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=True,
        )

        out2 = model.generate(
            input_ids.to(model.device),
            max_new_tokens=128,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
        )
        print("\nEXAONE reasoning output:\n", tokenizer.decode(out2[0]))