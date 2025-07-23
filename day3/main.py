from pprint import pprint
from models import llm, tavily_client
from document_loader import create_vectorstore, load_existing_vectorstore
from graders import question_router, retrieval_grader, rag_chain
from workflow import create_workflow

def main():
    """메인 실행 함수"""
    
    # 1. 기본 ChatOpenAI 호출
    # print("=== Basic ChatOpenAI Test ===")
    # print(llm.invoke("Hello, how are you?"))
    # print()
    
    # 2. Tavily 검색 테스트
    # print("=== Tavily Search Test ===")
    # response = tavily_client.search(query="Where does Messi play right now?", max_results=3)
    # context = [{"url": obj["url"], "content": obj["content"]} for obj in response['results']]
    
    # response_context = tavily_client.get_search_context(
    #     query="Where does Messi play right now?", 
    #     search_depth="advanced", 
    #     max_tokens=500
    # )
    
    # response_qna = tavily_client.qna_search(query="Where does Messi play right now?")
    # print()
    
    # 3. 문서 인덱싱 및 검색 테스트
    # print("=== Document Indexing and Retrieval Test ===")
    # retriever = create_vectorstore()
    
    # 4. 라우터 테스트
    # question = "What is prompt?"
    # docs = retriever.invoke(question)
    # print(question_router.invoke({"question": question}))
    
    # 5. 검색 평가기 테스트
    # doc_txt = docs[0].page_content
    # print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
    
    # 6. RAG 생성 테스트
    # generation = rag_chain.invoke({"context": docs, "question": question})
    # print(generation)
    
    # 7. 전체 워크플로우 실행
    print("\n=== Full Workflow Execution ===")
    app = create_workflow()
    
    inputs = {"question": "What is prompt?"}
    final_state = None
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
            final_state = value
    
    # 최종 결과 출력
    if final_state and "generation" in final_state:
        print("\n=== Final Answer ===")
        print(final_state["generation"])
    else:
        print("\n=== Final State ===")
        pprint(final_state)

if __name__ == "__main__":
    main() 