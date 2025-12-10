import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

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


# ----------------------------------------------------------
#  B) Gemini RAG → GPT final reasoning
#     (Not used now, but ready for later)
# ----------------------------------------------------------
def gemini_retrieve_chunks(question: str):
    """
    Retrieve top document chunks using File Search only.
    Used for GPT hybrid mode.
    """
    result = gemini.fileSearch.search({
        "fileSearchStoreName": FILE_SEARCH_STORE,
        "query": question
    })

    # Each document has: {"content": "..."}
    all_chunks = [doc["content"] for doc in result.get("documents", [])]
    return "\n\n".join(all_chunks)
