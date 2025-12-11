# create_filestore.py

# Used for the initial creation of File Search store for RAG with Gemini models
# ❌❌❌❌❌❌❌ DO NOT RUN THIS FILE MULTIPLE TIMES as it will create unncessary multiple stores ❌❌❌❌❌❌❌

#❌❌❌❌❌❌❌ 이 파일을 여러 번 실행하지 마세요. 불필요한 여러 스토어가 생성됩니다. ❌❌❌❌❌❌❌
"""
This script creates a filestore for document storage and retrieval.
문서 저장 및 검색을 위한 파일스토어를 생성하는 스크립트입니다.
"""
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types


# 1) Load .env and get GEMINI_API_KEY automatically
load_dotenv()
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

# 2) Create a File Search store
store = client.file_search_stores.create(
    config={"display_name": "shebots-rag-store"}
)
print("✅ File Search Store created:")
print("   name =", store.name)
# e.g. "fileSearchStores/shebots-rag-store-abc123"

# 3) (Optional) upload one small test file to confirm everything works
operation = client.file_search_stores.upload_to_file_search_store(
    file_search_store_name=store.name,
    file="/home/sslab/Documents/Nishtha/RAG_system/Location_Population_Data/location_query_result.txt",
    config={
        "display_name": "sample-doc",
        "chunking_config": {
            "white_space_config": {
                "max_tokens_per_chunk": 200,
                "max_overlap_tokens": 20,
            }
        },
    },
)

print("⏳ Uploading and indexing file...")
while not operation.done:
    time.sleep(2)
    operation = client.operations.get(operation)

print("✅ Upload completed!")

# 4) Test a query using File Search
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="이 문서의 내용을 한 줄로 요약해줘.",
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[store.name]
                )
            )
        ]
    ),
)

print("\n🧠 Model response:")
print(response.text)
print("\n📎 Grounding metadata (citations):")
print(response.candidates[0].grounding_metadata if response.candidates else None)
