from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
# from load_blogs import get_retriever # 관련성 평가에는 retriever가 필요 없음
from operator import itemgetter

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# JsonOutputParser 초기화
parser = JsonOutputParser()

# PromptTemplate 정의 (관련성 평가용)
prompt = PromptTemplate(
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

# 체인 구성 (관련성 평가용)
relevance_chain = prompt | llm | parser

# 샘플 쿼리 및 청크 정의
sample_user_query = "What is prompt engineering?"
sample_retrieved_chunk_relevant = "Prompt engineering is a discipline that focuses on developing and optimizing prompts to efficiently use language models for a wide range of applications and research topics."
sample_retrieved_chunk_irrelevant = "The capital of France is Paris."

# 관련성 평가 실행
print(f"Evaluating relevance for User Query: {sample_user_query}")
print(f"Retrieved Chunk (Relevant): {sample_retrieved_chunk_relevant}")
response_relevant = relevance_chain.invoke({"user_query": sample_user_query, "retrieved_chunk": sample_retrieved_chunk_relevant})
print(f"Relevance Result (Relevant): {response_relevant}")

print("\n") # 줄바꿈 추가

print(f"Retrieved Chunk (Irrelevant): {sample_retrieved_chunk_irrelevant}")
response_irrelevant = relevance_chain.invoke({"user_query": sample_user_query, "retrieved_chunk": sample_retrieved_chunk_irrelevant})
print(f"Relevance Result (Irrelevant): {response_irrelevant}")