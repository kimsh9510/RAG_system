import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing in .env")

client = genai.Client(api_key=api_key)

def list_file_stores():
    print("🔎 Listing File Search Stores...\n")

    pager = client.file_search_stores.list()

    found_any = False
    for i, store in enumerate(pager, start=1):
        found_any = True
        # store has name, display_name, create_time, etc.
        print(f"[{i}] name        : {store.name}")
        print(f"    display_name: {getattr(store, 'display_name', '(no display_name)')}")
        print("-" * 60)

    if not found_any:
        print("⚠ No File Search Stores found in this project / API key.")

if __name__ == "__main__":
    list_file_stores()
