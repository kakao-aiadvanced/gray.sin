from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
# from load_blogs import get_retriever # 관련성 평가에는 retriever가 필요 없음
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

# 샘플 쿼리 및 청크 정의
sample_user_query = "What is prompt engineering?"
sample_retrieved_chunk_relevant = "Prompt engineering is a discipline that focuses on developing and optimizing prompts to efficiently use language models for a wide range of applications and research topics."
sample_retrieved_chunk_irrelevant = "The capital of France is Paris."

print(f"Evaluating relevance for User Query: {sample_user_query}")

# 관련성 평가 및 답변 생성 (관련 있는 청크)
print(f"\n--- Relevant Chunk Test ---")
print(f"Retrieved Chunk: {sample_retrieved_chunk_relevant}")
relevance_result_relevant = relevance_chain.invoke({"user_query": sample_user_query, "retrieved_chunk": sample_retrieved_chunk_relevant})
print(f"Relevance Result: {relevance_result_relevant}")

if relevance_result_relevant.get('relevance') == 'yes':
    print("-> Chunk is relevant. Generating answer...")
    answer_response = answer_generation_chain.invoke({
        "user_query": sample_user_query,
        "context": sample_retrieved_chunk_relevant
    })
    print(f"Generated Answer: {answer_response}")
else:
    print("-> Chunk is not relevant. Skipping answer generation.")

# 관련성 평가 및 답변 생성 (관련 없는 청크)
print(f"\n--- Irrelevant Chunk Test ---")
print(f"Retrieved Chunk: {sample_retrieved_chunk_irrelevant}")
relevance_result_irrelevant = relevance_chain.invoke({"user_query": sample_user_query, "retrieved_chunk": sample_retrieved_chunk_irrelevant})
print(f"Relevance Result: {relevance_result_irrelevant}")

if relevance_result_irrelevant.get('relevance') == 'yes':
    print("-> Chunk is relevant. Generating answer...")
    answer_response = answer_generation_chain.invoke({
        "user_query": sample_user_query,
        "context": sample_retrieved_chunk_irrelevant
    })
    print(f"Generated Answer: {answer_response}")
else:
    print("-> Chunk is not relevant. Skipping answer generation.")
