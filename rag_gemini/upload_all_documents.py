import os

import requests
from dotenv import load_dotenv
from data import FILES_TO_UPLOAD

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
STORE_NAME = os.getenv("FILE_SEARCH_STORE_NAME")  # e.g. "fileSearchStores/your-store-id"

if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY missing")

if not STORE_NAME:
    raise ValueError("❌ FILE_SEARCH_STORE_NAME missing")

MIME_TYPES = {
    ".txt": "text/plain",
    ".pdf": "application/pdf",
}

def get_mime_type(filename):
    ext = os.path.splitext(filename)[1].lower()
    return MIME_TYPES.get(ext, "application/octet-stream")


def upload_file(path: str):
    print(f"\n📤 Uploading: {path}")
    mime_type = get_mime_type(path)
    url = f"https://generativelanguage.googleapis.com/upload/v1beta/{STORE_NAME}:uploadToFileSearchStore?key={API_KEY}"
    try:
        with open(path, "rb") as f:
            files = {"file": (os.path.basename(path), f, mime_type)}
            data = {"mimeType": mime_type}
            response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            print(f"   ✅ SUCCESS: {os.path.basename(path)}")
            return True
        else:
            print(f"❌ FAILED: {path}")
            print("Reason:", response.text)
            return False
    except Exception as e:
        print(f"❌ FAILED: {path}")
        print("Reason:", e)
        return False


def upload_all(file_list):
    uploaded = 0
    failed = 0

    for path in file_list:
        if not os.path.isfile(path):
            print(f"⚠ Skipping (not found): {path}")
            failed += 1
            continue
        if upload_file(path):
            uploaded += 1
        else:
            failed += 1

    print("\n=== UPLOAD SUMMARY ===")
    print("Total files    :", len(file_list))
    print("Uploaded       :", uploaded)
    print("Failed / skipped:", failed)
    print("======================")

if __name__ == "__main__":
    upload_all(FILES_TO_UPLOAD)
