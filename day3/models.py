from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tavily import TavilyClient
from config import LLM_MODEL, LLM_TEMPERATURE, EMBEDDING_MODEL, TAVILY_API_KEY

# LLM 초기화
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Tavily 클라이언트 초기화
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) 