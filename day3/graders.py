from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from models import llm

# 질문 라우터
router_system = """You are an expert at routing a user question to a vectorstore or web search.
Use the vectorstore for questions on agent, LLM agents, prompt engineering, and adversarial attacks.
You do not need to be stringent with the keywords in the question related to these topics.
Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
Return the a JSON with a single key 'datasource' and no premable or explanation. Question to route"""

router_prompt = ChatPromptTemplate.from_messages([
    ("system", router_system),
    ("human", "question: {question}"),
])

question_router = router_prompt | llm | JsonOutputParser()

# 검색 문서 평가기
retrieval_system = """You are a grader assessing relevance
of a retrieved document to a user question. If the document contains keywords related to the user question,
grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
Provide the binary score as a JSON with a single key 'score' and no premable or explanation."""

retrieval_prompt = ChatPromptTemplate.from_messages([
    ("system", retrieval_system),
    ("human", "question: {question}\n\n document: {document} "),
])

retrieval_grader = retrieval_prompt | llm | JsonOutputParser()

# RAG 답변 생성기
rag_system = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system),
    ("human", "question: {question}\n\n context: {context} "),
])

rag_chain = rag_prompt | llm | StrOutputParser()

# 할루시네이션 평가기
hallucination_system = """You are a grader assessing whether
an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
single key 'score' and no preamble or explanation."""

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", hallucination_system),
    ("human", "documents: {documents}\n\n answer: {generation} "),
])

hallucination_grader = hallucination_prompt | llm | JsonOutputParser()

# 답변 평가기
answer_system = """You are a grader assessing whether an
answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", answer_system),
    ("human", "question: {question}\n\n answer: {generation} "),
])

answer_grader = answer_prompt | llm | JsonOutputParser() 