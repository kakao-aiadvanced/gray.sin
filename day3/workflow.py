from typing import List
from langchain_core.documents import Document
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from nodes import (
    web_search, retrieve, grade_documents, generate,
    route_question, decide_to_generate, grade_generation_v_documents_and_question
)

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]

def create_workflow():
    """RAG 워크플로우를 생성합니다."""
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("websearch", web_search)  # web search
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate
    workflow.add_node("grade_generation", grade_generation)  # grade generation v documents and question

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "yes": "generate",
            "no": "websearch",
        },
    )
    workflow.add_edge("websearch", "grade_documents")
    # TODO 최대 1회 로직 추가

    workflow.add_edge("generate", "grade_generation")
    workflow.add_conditional_edges(
        "grade_generation",
        grade_generation_v_documents_and_question,
        {
            "yes": END,
            "no": "generate",
        },
    )


    # Compile
    return workflow.compile() 