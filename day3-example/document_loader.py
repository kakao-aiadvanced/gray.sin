from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import embeddings
from config import URLS, CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME

def load_and_index_documents():
    """웹 문서들을 로드하고 벡터스토어에 인덱싱합니다."""
    # 문서 로드
    docs = [WebBaseLoader(url).load() for url in URLS]
    docs_list = [item for sublist in docs for item in sublist]

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # 벡터스토어에 추가
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    
    return vectorstore.as_retriever() 