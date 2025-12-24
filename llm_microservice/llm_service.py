"""
llm_service.py

A minimal FastAPI server that exposes the LLM node logic as an HTTP API.
This wraps the llm_node logic from nodes.py and loads the model as in models.py.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, Request
from pydantic import BaseModel
from models import load_llama3, load_paid_gpt
from nodes import llm_node, State

# Load the LLM (same as in your main app)
llm = load_paid_gpt()
llm_node_fn = llm_node(llm)

app = FastAPI()

class LLMRequest(BaseModel):
    query: str
    location_si: str = ""
    location_gu: str = ""
    location_dong: str = ""
    disaster: str = ""
    law_ctx: str = ""
    law_flooding_ctx: str = ""
    law_blackout_ctx: str = ""
    manual_ctx: str = ""
    basic_ctx: str = ""
    past_ctx: str = ""
    population_ctx: str = ""

@app.post("/generate")
def generate(request: LLMRequest):
    # Build the state dict as expected by llm_node
    state = request.dict()
    result = llm_node_fn(state)
    return {"answer": result["answer"]}
