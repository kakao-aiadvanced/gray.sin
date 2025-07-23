from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# API 키들
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 모델 설정
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_TEMPERATURE = 0

# 문서 설정
URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

CHUNK_SIZE = 250
CHUNK_OVERLAP = 0
COLLECTION_NAME = "rag-chroma" 