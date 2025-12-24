"""
client_example.py

Example client code to call the LLM microservice from your main LangGraph app.
Replace the direct llm_node call with an HTTP request to the microservice.
"""
import requests

def call_llm_microservice(state: dict, url: str = "http://localhost:8080/generate"):
    response = requests.post(url, json=state)
    response.raise_for_status()
    return response.json()["answer"]

# Example usage:
if __name__ == "__main__":
    state = {
        "query": "침수 시 연계 재난?",
        "location_si": "서울특별시",
        "location_gu": "서초구",
        "location_dong": "방배4동"
    }
    print(call_llm_microservice(state))
