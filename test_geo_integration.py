"""
Test script to verify geospatial data integration with RAG system
"""
import sys
import os
from langgraph.graph import StateGraph, START, END
from knowledge_base_copy1 import build_vectorstores
from models import load_EXAONE_api
from nodes import State, retrieval_law_node, retrieval_manual_node, retrieval_basic_node, retrieval_past_node, llm_node, response_node

print("=" * 60)
print("Testing Geospatial Data Integration with RAG")
print("=" * 60)

# Build vector stores (includes geospatial data)
print("\n1. Building vector stores with geospatial data...")
try:
    vectordb_law, vectordb_manual, vectordb_basic, vectordb_past = build_vectorstores()
    print("✓ Vector stores built successfully")
except Exception as e:
    print(f"✗ Error building vector stores: {e}")
    sys.exit(1)

# Test geospatial document retrieval
print("\n2. Testing geospatial document retrieval...")
try:
    # Search for location-specific information
    test_queries = [
        "서울 강남구 인구 밀도",
        "부산 해운대구 지역 특성",
        "인천 지역 재난 대응"
    ]
    
    for query in test_queries:
        print(f"\n   Query: {query}")
        docs = vectordb_basic.similarity_search(query, k=3)
        geo_docs = [d for d in docs if d.metadata.get("type") == "geospatial"]
        
        if geo_docs:
            print(f"   ✓ Found {len(geo_docs)} geospatial document(s)")
            print(f"   Sample: {geo_docs[0].metadata.get('area_name', 'N/A')}")
            # Show first 200 characters of content
            content_preview = geo_docs[0].page_content[:200]
            print(f"   Content preview: {content_preview}...")
        else:
            print(f"   ✗ No geospatial documents found")
    
    print("\n✓ Geospatial retrieval test completed")
except Exception as e:
    print(f"✗ Error during retrieval test: {e}")
    import traceback
    traceback.print_exc()

# Test full RAG pipeline (optional - comment out if LLM not available)
print("\n3. Testing full RAG pipeline...")
print("   (Skipping LLM test - uncomment to run full test)")
"""
try:
    llm = load_EXAONE_api()
    
    graph = StateGraph(State)
    graph.add_node("retrieval_law", retrieval_law_node(vectordb_law))
    graph.add_node("retrieval_manual", retrieval_manual_node(vectordb_manual))
    graph.add_node("retrieval_basic", retrieval_basic_node(vectordb_basic))
    graph.add_node("retrieval_past", retrieval_past_node(vectordb_past))
    graph.add_node("llm", llm_node(llm))
    graph.add_node("response", response_node)
    
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
    
    app = graph.compile()
    result = app.invoke({
        "query": "서울 강남구에서 태풍 발생 시 예상되는 피해 규모와 대응 방안"
    })
    
    print("✓ Full pipeline test completed")
except Exception as e:
    print(f"✗ Error in full pipeline: {e}")
"""

print("\n" + "=" * 60)
print("Integration Test Complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Run main.py to test with actual disaster scenarios")
print("2. Check if geographic context appears in responses")
print("3. Verify population density and area data is used in risk assessment")
