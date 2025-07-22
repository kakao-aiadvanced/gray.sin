from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from load_blogs import get_retriever  # 실제 retriever 사용
from operator import itemgetter

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# JsonOutputParser 초기화
parser = JsonOutputParser()

# 1. 관련성 평가 PromptTemplate 정의
relevance_prompt = PromptTemplate(
    template="""You are an expert in evaluating the relevance between a user query and a retrieved document chunk.
Your task is to determine if the retrieved chunk is relevant to the user's query.
Output your answer in JSON format, using the following structure:
{{"relevance": "yes"}} if the chunk is relevant, or {{"relevance": "no"}} if it is not.

User Query: {user_query}
Retrieved Chunk: {retrieved_chunk}
{format_instructions}
""",
    input_variables=["user_query", "retrieved_chunk"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

# 관련성 평가 체인
relevance_chain = relevance_prompt | llm | parser

# 2. 답변 생성 PromptTemplate 정의
answer_generation_prompt = PromptTemplate(
    template="""Answer the user query based on the provided context.
If the context does not contain enough information to answer the query, state that you cannot answer based on the provided context.
Context: {context}
User Query: {user_query}
Output your answer in JSON format, using the following structure: {{"answer": "Your answer here"}}.
{format_instructions}
""",
    input_variables=["user_query", "context"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

# 답변 생성 체인
answer_generation_chain = answer_generation_prompt | llm | parser

# 3. Hallucination 평가 PromptTemplate 정의
hallucination_prompt = PromptTemplate(
    template="""You are an expert in evaluating whether an AI-generated answer contains hallucinations.
Your task is to determine if the generated answer is faithful to the provided context and does not contain any information that is not supported by the context.

A hallucination occurs when:
1. The answer contains information that is not present in the provided context
2. The answer contradicts information in the context
3. The answer makes up facts, numbers, or details not found in the context
4. The answer draws conclusions that go beyond what the context supports

Evaluate the generated answer against the provided context and determine if there are any hallucinations.
Output your answer in JSON format, using the following structure:
{{"hallucination": "yes"}} if the answer contains hallucinations, or {{"hallucination": "no"}} if it does not.

Context: {context}
Generated Answer: {generated_answer}
{format_instructions}
""",
    input_variables=["context", "generated_answer"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

# Hallucination 평가 체인
hallucination_chain = hallucination_prompt | llm | parser

def query_rag_system(user_query):
    """RAG 시스템으로 사용자 쿼리에 대한 답변을 생성하는 함수"""
    # retriever 초기화
    print("벡터 저장소 초기화 중...")
    retriever = get_retriever()
    
    # 사용자 쿼리로 관련 문서들 검색
    print(f"\n사용자 쿼리: {user_query}")
    print("관련 문서 검색 중...")
    retrieved_docs = retriever.invoke(user_query)
    
    print(f"검색된 문서 개수: {len(retrieved_docs)}")
    
    relevant_chunks = []
    
    # 각 검색된 문서에 대해 관련성 평가
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- 문서 {i+1} 관련성 평가 ---")
        chunk_content = doc.page_content
        print(f"문서 내용 미리보기: {chunk_content[:100]}...")
        
        # 관련성 평가
        relevance_result = relevance_chain.invoke({
            "user_query": user_query, 
            "retrieved_chunk": chunk_content
        })
        print(f"관련성 평가 결과: {relevance_result}")
        
        if relevance_result.get('relevance') == 'yes':
            relevant_chunks.append(chunk_content)
            print("-> 관련성 있음. 컨텍스트에 추가됨.")
        else:
            print("-> 관련성 없음. 제외됨.")
    
    # 관련성 있는 청크들로 답변 생성
    if relevant_chunks:
        print(f"\n--- 답변 생성 ---")
        print(f"관련성 있는 문서 개수: {len(relevant_chunks)}")
        
        # 모든 관련 청크를 하나의 컨텍스트로 결합
        combined_context = "\n\n".join(relevant_chunks)
        
        # 답변 생성 및 Hallucination 검증 (최대 2회 시도)
        max_attempts = 2
        attempt = 1
        
        while attempt <= max_attempts:
            print(f"\n--- 답변 생성 (시도 {attempt}/{max_attempts}) ---")
            answer_response = answer_generation_chain.invoke({
                "user_query": user_query,
                "context": combined_context
            })
            print(f"생성된 답변: {answer_response}")
            
            # Hallucination 평가
            print(f"\n--- Hallucination 평가 (시도 {attempt}) ---")
            hallucination_result = hallucination_chain.invoke({
                "context": combined_context,
                "generated_answer": answer_response.get('answer', '')
            })
            print(f"Hallucination 평가 결과: {hallucination_result}")
            
            if hallucination_result.get('hallucination') == 'no':
                print("✅ Hallucination이 감지되지 않았습니다.")
                break
            else:
                print("⚠️  경고: 생성된 답변에 Hallucination이 감지되었습니다!")
                if attempt < max_attempts:
                    print("🔄 답변을 재생성합니다...")
                    attempt += 1
                else:
                    print("📝 최대 시도 횟수 도달. 현재 답변을 제공합니다.")
                    break
        
        # 최종 답변과 출처 정보 제공
        print(f"\n--- 최종 답변 및 출처 ---")
        print(f"답변: {answer_response.get('answer', '')}")
        
        # 출처 정보 추출 및 표시
        print(f"\n📚 출처 정보:")
        for i, doc in enumerate([doc for doc in retrieved_docs if doc.page_content in relevant_chunks]):
            source_info = f"출처 {i+1}: "
            if hasattr(doc, 'metadata') and doc.metadata:
                if 'source' in doc.metadata:
                    source_info += f"{doc.metadata['source']}"
                if 'title' in doc.metadata:
                    source_info += f" - {doc.metadata['title']}"
            else:
                source_info += f"문서 내용: {doc.page_content[:100]}..."
            print(source_info)
        
        return answer_response
    else:
        print("\n관련성 있는 문서가 없습니다.")
        return {"answer": "제공된 문서들에서 해당 질문에 대한 답변을 찾을 수 없습니다."}

# 샘플 실행
if __name__ == "__main__":
    # 테스트용 샘플 쿼리들 (관련성 있는 것과 없는 것 포함)
    sample_queries = [
        # AI/ML 관련 (관련성 있음) - 출처 정보와 재시도 로직 테스트
        "What is prompt engineering?",
        
        # 완전히 다른 주제 (관련성 없음) - 관련성 필터링 테스트
        "What is the capital of France?"
    ]
    
    for query in sample_queries:
        print("=" * 80)
        result = query_rag_system(query)
        print("=" * 80)
        print()
