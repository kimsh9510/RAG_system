import sys
import os

#백엔드 - fastapi 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
import mimetypes

from langgraph.graph import StateGraph, START, END
from knowledge_base_copy1 import build_vectorstores
from models import load_qwen, load_solar_pro, load_solar_pro2, load_llama3, load_EXAONE, load_gpt, load_paid_gpt
from nodes import State, retrieval_law_node, retrieval_flooding_law_node, retrieval_blackout_law_node, retrieval_manual_node, retrieval_basic_node, retrieval_past_node, retrieval_gis_node, llm_node, response_node

vectordb_law, vectordb_flooding_law, vectordb_blackout_law, vectordb_manual, vectordb_basic, vectordb_past, vectordb_gis = build_vectorstores()

# LLM 모델 로드 (Llama + GPT)
print("AI 모델 로딩 중 (시간이 조금 걸릴 수 있습니다)...")

# Llama3
try:
    llama_model = load_llama3()
    print("LLama 3 로드 완료")
except Exception as e:
    print(f"LLama 3 로드 실패: {e}")
    llama_model = None

#GPT-4
# .env 파일에 OPENAI_API_KEY 또는 GPT_API_KEY가 있어야 합니다.
try:
    gpt_model = load_paid_gpt() # 또는 gpt-4o-mini
    print("gpt-5.1 연결 완료")
except Exception as e:
    print(f"gpt-5.1 로드 실패 (API 키 확인 필요): {e}")
    gpt_model = None

# (3) 모델 딕셔너리 생성 (nodes.py의 llm_node로 전달됨)
models_map = {
    "llama3": llama_model,
    "gpt-5.1": gpt_model
}



def route_disaster(state):
    """LangGraph의 State에서 disaster 값을 읽어 다음 법령 노드를 결정합니다."""
    disaster_type = state.get("disaster", "default")
    
    if disaster_type == "침수":
        return "flooding_law_node"
    elif disaster_type == "정전":
        return "blackout_law_node"
    else:
        # 법령 노드로 라우팅
        return "retrieval_law" 

# 4. LangGraph 정의
def build_graph():
    graph = StateGraph(State)
    
    #항상 실행
    graph.add_node("retrieval_manual", retrieval_manual_node(vectordb_manual))
    graph.add_node("retrieval_basic", retrieval_basic_node(vectordb_basic))
    graph.add_node("retrieval_past", retrieval_past_node(vectordb_past))
    graph.add_node("retrieval_gis", retrieval_gis_node(vectordb_gis))  
    
    # 조건부로 실행
    graph.add_node("retrieval_law", retrieval_law_node(vectordb_law))
    graph.add_node("flooding_law_node", retrieval_flooding_law_node(vectordb_flooding_law))
    graph.add_node("blackout_law_node", retrieval_blackout_law_node(vectordb_blackout_law))

    graph.add_node("llm", llm_node(models_map))
    graph.add_node("response", response_node)
    
    graph.add_conditional_edges(
        START,
        route_disaster, 
        {
            "flooding_law_node": "flooding_law_node", # '침수'인 경우
            "blackout_law_node": "blackout_law_node", # '정전'인 경우
            "retrieval_law": "retrieval_law",         # 그 외의 경우
        }
    )

    graph.add_edge(START, "retrieval_manual")
    graph.add_edge(START, "retrieval_basic")
    graph.add_edge(START, "retrieval_past")
    graph.add_edge(START, "retrieval_gis")  

    all_retrieval_nodes = [
        "retrieval_law", "retrieval_manual", "retrieval_basic", "retrieval_past",
        "flooding_law_node", "blackout_law_node", "retrieval_gis" 
    ]
    
    for node in all_retrieval_nodes:
        graph.add_edge(node, "llm")

    graph.add_edge("llm", "response")
    graph.add_edge("response", END)
    
    return graph.compile()


LLM_APP=build_graph()
app= FastAPI()

#CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    location_si: str
    location_gu: str
    location_dong : str
    disaster: str
    model:str="llama3"

@app.post("/")
@app.post("/chat")
async def chat_with_llm(request_data: ChatRequest):
    try:
        input_state = request_data.model_dump()

        input_state.update({
            # 프론트의 'model' 값을 State의 'selected_model'로 매핑
            "selected_model": request_data.model,
            
            # (초기화)
            "law_ctx": "",
            "law_flooding_ctx": "",
            "law_blackout_ctx": "",
            "manual_ctx": "",
            "basic_ctx": "",
            "past_ctx": "",
            "gis_ctx": "",
            "answer": ""
        })
        
        result=LLM_APP.invoke(input_state)
        final_answer=result.get('answer','Langgraph에서 최종응답을 찾을 수 없습니다.')
        
        return {"status":"success","response":final_answer}
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return {"status": "error", "message": f"LLM 처리 중 오류가 발생했습니다: {e}"}

# 윈도우 환경 MIME 타입 이슈 방지
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")



if getattr(sys, 'frozen', False):
    # PyInstaller로 실행된 경우 (임시 경로 사용)
    BASE_DIR = Path(sys._MEIPASS)
else:
    # 일반 파이썬으로 실행된 경우 (현재 파일 경로 사용)
    BASE_DIR = Path(__file__).resolve().parent
    
FRONTEND_DIR = BASE_DIR / "dist"
ASSETS_DIR = FRONTEND_DIR / "assets"

if FRONTEND_DIR.exists():
    
    # assets 경로를 'StaticFiles'로 직접 연결
    if ASSETS_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
        print("Assets 폴더 마운트 성공")
    else:
        print("주의: dist 폴더 안에 assets 폴더가 없습니다.")

    @app.get("/")
    async def read_root():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"error": "index.html 파일이 없습니다."}

    @app.get("/{full_path:path}")
    async def read_frontend(full_path: str):
        # API 요청은 건너뜀 - 파일 없으면 바로 html 주는 걸 방지
        if full_path.startswith("api") or full_path.startswith("chat"):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})

        # 만약 assets 파일인데 위에서 마운트가 안 돼서 여기로 왔다 -> 404 에러 리턴
        if full_path.startswith("assets/"):
             return JSONResponse(status_code=404, content={"detail": "Asset file missing"})

        # 3. 실제 파일이 존재하면 그 파일 제공 - 진짜 존재하는 파일(물건)과 가짜 주소(React 라우팅)를 구분
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        
        # 4. 그 외에는 전부 index.html 반환 
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
            
        return {"error": "프론트엔드 파일이 없습니다."}

else:
    print(f" 오류: '{FRONTEND_DIR}' 폴더를 찾을 수 없습니다.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)