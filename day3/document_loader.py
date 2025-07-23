from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import embeddings
from config import URLS, CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME
import os

def create_vectorstore():
    """웹 문서들을 로드하고 새로운 벡터스토어를 생성합니다."""
    print("새로운 벡터스토어 생성 중...")
    
    # 문서 로드
    docs = [WebBaseLoader(url).load() for url in URLS]
    docs_list = [item for sublist in docs for item in sublist]

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # 벡터스토어에 추가 (디스크에 저장)
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print(f"벡터스토어 생성 완료: {COLLECTION_NAME}")
    return vectorstore.as_retriever()

def load_existing_vectorstore():
    """기존 벡터스토어를 로드합니다. 없으면 에러를 발생시킵니다."""
    try:
        # 기존 벡터스토어 로드 시도 (디스크에서)
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        
        # 컬렉션이 실제로 존재하고 문서가 있는지 확인
        collection = vectorstore._collection
        if collection.count() == 0:
            raise ValueError(f"벡터스토어 '{COLLECTION_NAME}' 컬렉션이 비어있습니다.")
        
        print(f"기존 벡터스토어 로드 완료: {COLLECTION_NAME} (문서 수: {collection.count()})")
        return vectorstore.as_retriever()
        
    except Exception as e:
        raise RuntimeError(
            f"기존 벡터스토어를 로드할 수 없습니다: {e}\n"
            f"먼저 main.py를 실행하여 벡터스토어를 생성해주세요."
        ) 