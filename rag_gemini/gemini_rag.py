import os
import time
import requests
from google import genai
from dotenv import load_dotenv

load_dotenv()

"""
This module implements the Gemini RAG (Retrieval-Augmented Generation) logic.
이 모듈은 Gemini RAG(검색 기반 생성) 로직을 구현합니다.
"""
# Load API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("❌ GEMINI_API_KEY missing in .env")

gemini = genai.Client(api_key=api_key)

# Load File Search Store
FILE_SEARCH_STORE = os.getenv("FILE_SEARCH_STORE_NAME")
if not FILE_SEARCH_STORE:
    raise RuntimeError("❌ FILE_SEARCH_STORE_NAME missing in .env")


# ----------------------------------------------------------
#  A) GEMINI-ONLY RAG FUNCTION
# ----------------------------------------------------------
def gemini_rag_answer(question: str):
    """
    Gemini-only RAG (simplest mode)
    Gemini performs:
    - Retrieval from File Search
    - Answer generation with grounding
    """
    response = gemini.models.generate_content(
    model="gemini-2.5-flash",
    contents=question,
    config={
        "tools": [{
            "fileSearch": {
                "fileSearchStoreNames": [FILE_SEARCH_STORE]
            }
        }]
    })

    return response.text