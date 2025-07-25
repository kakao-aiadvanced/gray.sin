from models import tavily_client
from graders import retrieval_grader, rag_chain, hallucination_grader, answer_grader, generate_decision_grader

def web_search(state):
    """
    Web search based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """
    print("---WEB SEARCH---")
    question = state["question"]
    print({"question": question})

    # Web search - 구조화된 응답을 받아서 URL 정보 보존
    response = tavily_client.search(
        query=question, search_depth="advanced", max_results=3
    )
    
    # Create document-like structure with source information
    from langchain_core.documents import Document
    docs = []
    
    for result in response['results']:
        doc = Document(
            page_content=result['content'],
            metadata={
                'source': result['url'],
                'title': result['title'],
                'score': result['score'],
                'source_type': 'web_search'
            }
        )
        docs.append(doc)
    
    print(f"웹 검색 결과: {len(docs)}개 문서 찾음")
    for doc in docs:
        print(f"  - {doc.metadata['title']}: {doc.metadata['source']}")
    
    return {"documents": docs, "question": question}

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Import retriever here to avoid circular imports
    from document_loader import load_existing_vectorstore
    retriever = load_existing_vectorstore()
    
    # Retrieval
    documents = retriever.invoke(question)
    
    # 벡터스토어 문서에 source_type 메타데이터 추가
    for doc in documents:
        if 'source_type' not in doc.metadata:
            doc.metadata['source_type'] = 'vector_store'
    
    print(f"벡터 검색 결과: {len(documents)}개 문서 찾음")
    for doc in documents:
        source = doc.metadata.get('source', '알 수 없음')
        title = doc.metadata.get('title', '제목 없음')
        print(f"  - {title}: {source}")
    
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    
    # 출처 정보 추가
    sources = format_sources(documents)
    
    # 생성된 답변에 출처 정보 추가
    full_response = f"{generation}\n\n📚 **출처:**\n" + "\n".join(sources)
    
    return {"documents": documents, "question": question, "generation": full_response}

def format_sources(documents):
    """문서들의 출처 정보를 포맷팅"""
    sources = []
    seen_sources = set()  # 중복 제거용
    
    for i, doc in enumerate(documents, 1):
        source_url = doc.metadata.get('source', '')
        title = doc.metadata.get('title', f'문서 {i}')
        source_type = doc.metadata.get('source_type', '알 수 없음')
        
        # 중복 URL 제거
        if source_url and source_url not in seen_sources:
            seen_sources.add(source_url)
            if source_type == 'web_search':
                score = doc.metadata.get('score', 0)
                sources.append(f"{len(sources)+1}. **{title}** (웹 검색, 신뢰도: {score:.2f})\n   🔗 {source_url}")
            else:
                sources.append(f"{len(sources)+1}. **{title}** (벡터 데이터베이스)\n   🔗 {source_url}")
    
    return sources

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    relevanceCheckCount = state.get("relevanceCheckCount", 0)

    # 웹 검색 후 최대 2번까지 관련성 검사를 허용 (벡터 검색 1번 + 웹 검색 후 1번)
    if relevanceCheckCount >= 2:
        print("---DECISION: MAX RELEVANCE CHECK COUNT REACHED, INCLUDE WEB SEARCH---")
        raise Exception("failed: not relevant")

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            continue

    return {"documents": filtered_docs, "question": question, "relevanceCheckCount": relevanceCheckCount + 1}

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    
    from graders import question_router
    source = question_router.invoke({"question": question})
    print(source)
    print(source["datasource"])
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer using LLM evaluation of question-document relevance

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision "yes" or "no"
    """
    print("---ASSESS DOCUMENTS FOR GENERATION---")
    question = state["question"]
    documents = state["documents"]
    
    # documents가 비어있으면 no, 있으면 yes를 반환한다.
    if len(documents) == 0:
        print("---DECISION: NO DOCUMENTS FOUND, INCLUDE WEB SEARCH---")
        return "no"
    else:
        print("---DECISION: DOCUMENTS FOUND, NO WEB SEARCH---")
        return "yes"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    hasHallucination = state.get("hasHallucination", False)
    hallucinationCheckCount = state.get("hallucinationCheckCount", 0)

    # 할루시네이션 체크를 최대 2번까지 허용
    if hallucinationCheckCount >= 2:
        print("---DECISION: MAX HALLUCINATION CHECK COUNT REACHED, INCLUDE WEB SEARCH---")
        raise Exception("failed: not hallucination")

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return {
            "question": question,
            "documents": documents,
            "generation": generation,
            "hasHallucination": False, 
            "hallucinationCheckCount": hallucinationCheckCount
        }
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return {
            "question": question,
            "documents": documents,
            "generation": generation,
            "hasHallucination": True, 
            "hallucinationCheckCount": hallucinationCheckCount + 1
        }

def decide_to_print(state):
    """
    Determines whether to end the workflow
    """
    # hasHallucination이 True이면 출력하지 않고 다시 생성한다.
    # hasHallucination이 False이면 출력한다.
    hasHallucination = state.get("hasHallucination", False)
    if hasHallucination:
        print("---DECISION: HAS HALLUCINATION, RE-GENERATE---")
        return "no"
    else:
        print("---DECISION: NO HALLUCINATION, PRINT ANSWER---")
        return "yes"