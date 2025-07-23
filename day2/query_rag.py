from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from load_blogs import get_retriever
from operator import itemgetter
from typing import List, Dict, Any, Tuple


class RAGSystemConfig:
    """RAG 시스템 설정"""
    def __init__(self):
        self.retrieval_k = 6
        self.max_attempts = 2
        self.llm_model = "gpt-4o-mini"
        self.llm_temperature = 0


class RAGSystem:
    """RAG (Retrieval-Augmented Generation) 시스템"""
    
    def __init__(self, config: RAGSystemConfig = None):
        """RAG 시스템 초기화"""
        self.config = config or RAGSystemConfig()
        
        # LLM 및 파서 초기화
        self.llm = ChatOpenAI(
            model=self.config.llm_model, 
            temperature=self.config.llm_temperature
        )
        self.parser = JsonOutputParser()
        
        # 체인들 초기화
        self._setup_chains()
        
        # Retriever 초기화 (한 번만!)
        print("벡터 저장소 초기화 중...")
        self.retriever = get_retriever()
        print("RAG 시스템 초기화 완료!")
    
    def _setup_chains(self):
        """프롬프트 체인들 설정"""
        # 1. 관련성 평가 체인
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
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.relevance_chain = relevance_prompt | self.llm | self.parser
        
        # 2. 답변 생성 체인
        answer_prompt = PromptTemplate(
            template="""Answer the user query based on the provided context.
If the context does not contain enough information to answer the query, state that you cannot answer based on the provided context.
Context: {context}
User Query: {user_query}
Output your answer in JSON format, using the following structure: {{"answer": "Your answer here"}}.
{format_instructions}
""",
            input_variables=["user_query", "context"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.answer_chain = answer_prompt | self.llm | self.parser
        
        # 3. Hallucination 평가 체인
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
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.hallucination_chain = hallucination_prompt | self.llm | self.parser
    
    def retrieve_documents(self, query: str) -> List[Any]:
        """문서 검색"""
        print(f"사용자 쿼리: {query}")
        print("관련 문서 검색 중...")
        retrieved_docs = self.retriever.invoke(query)
        print(f"검색된 문서 개수: {len(retrieved_docs)}")
        return retrieved_docs
    
    def evaluate_relevance(self, documents: List[Any], query: str) -> Tuple[List[str], List[Any]]:
        """문서 관련성 평가"""
        relevant_chunks = []
        relevant_docs = []
        
        for i, doc in enumerate(documents):
            print(f"\n--- 문서 {i+1} 관련성 평가 ---")
            chunk_content = doc.page_content
            print(f"문서 내용 미리보기: {chunk_content[:100]}...")
            
            relevance_result = self.relevance_chain.invoke({
                "user_query": query, 
                "retrieved_chunk": chunk_content
            })
            print(f"관련성 평가 결과: {relevance_result}")
            
            if relevance_result.get('relevance') == 'yes':
                relevant_chunks.append(chunk_content)
                relevant_docs.append(doc)
                print("-> 관련성 있음. 컨텍스트에 추가됨.")
            else:
                print("-> 관련성 없음. 제외됨.")
        
        return relevant_chunks, relevant_docs
    
    def generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        """답변 생성"""
        return self.answer_chain.invoke({
            "user_query": query,
            "context": context
        })
    
    def check_hallucination(self, answer: str, context: str) -> Dict[str, Any]:
        """Hallucination 검사"""
        return self.hallucination_chain.invoke({
            "context": context,
            "generated_answer": answer
        })
    
    def generate_answer_with_validation(self, query: str, context: str) -> Dict[str, Any]:
        """검증과 함께 답변 생성 (재시도 로직 포함)"""
        attempt = 1
        
        while attempt <= self.config.max_attempts:
            print(f"\n--- 답변 생성 (시도 {attempt}/{self.config.max_attempts}) ---")
            answer_response = self.generate_answer(query, context)
            print(f"생성된 답변: {answer_response}")
            
            # Hallucination 평가
            print(f"\n--- Hallucination 평가 (시도 {attempt}) ---")
            hallucination_result = self.check_hallucination(
                answer_response.get('answer', ''), 
                context
            )
            print(f"Hallucination 평가 결과: {hallucination_result}")
            
            if hallucination_result.get('hallucination') == 'no':
                print("✅ Hallucination이 감지되지 않았습니다.")
                break
            else:
                print("⚠️  경고: 생성된 답변에 Hallucination이 감지되었습니다!")
                if attempt < self.config.max_attempts:
                    print("🔄 답변을 재생성합니다...")
                    attempt += 1
                else:
                    print("📝 최대 시도 횟수 도달. 현재 답변을 제공합니다.")
                    break
        
        return answer_response
    
    def format_sources(self, relevant_docs: List[Any]) -> List[str]:
        """출처 정보 포맷팅"""
        sources = []
        for i, doc in enumerate(relevant_docs):
            source_info = f"출처 {i+1}: "
            if hasattr(doc, 'metadata') and doc.metadata:
                if 'source' in doc.metadata:
                    source_info += f"{doc.metadata['source']}"
                if 'title' in doc.metadata:
                    source_info += f" - {doc.metadata['title']}"
            else:
                source_info += f"문서 내용: {doc.page_content[:100]}..."
            sources.append(source_info)
        return sources
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """RAG 시스템 메인 쿼리 메서드"""
        print("\n" + "="*80)
        
        # 1. 문서 검색
        retrieved_docs = self.retrieve_documents(user_query)
        
        # 2. 관련성 평가
        relevant_chunks, relevant_docs = self.evaluate_relevance(retrieved_docs, user_query)
        
        if not relevant_chunks:
            print("\n관련성 있는 문서가 없습니다.")
            return {"answer": "제공된 문서들에서 해당 질문에 대한 답변을 찾을 수 없습니다."}
        
        # 3. 답변 생성
        print(f"\n--- 답변 생성 ---")
        print(f"관련성 있는 문서 개수: {len(relevant_chunks)}")
        
        combined_context = "\n\n".join(relevant_chunks)
        answer_response = self.generate_answer_with_validation(user_query, combined_context)
        
        # 4. 최종 결과 출력
        print(f"\n--- 최종 답변 및 출처 ---")
        print(f"답변: {answer_response.get('answer', '')}")
        
        # 출처 정보 표시
        print(f"\n📚 출처 정보:")
        sources = self.format_sources(relevant_docs)
        for source in sources:
            print(source)
        
        print("="*80)
        return answer_response


# 샘플 실행
if __name__ == "__main__":
    # RAG 시스템 초기화
    rag_system = RAGSystem()
    
    # 테스트용 샘플 쿼리들
    sample_queries = [
        "What is prompt engineering?",
        "What is the capital of France?"
    ]
    
    for query in sample_queries:
        result = rag_system.query(query)
        print()
