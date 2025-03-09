"""
Node for retrieving relevant documents from the vector store.
"""
import logging
from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever

logger = logging.getLogger("self_rag.retrieve")


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve relevant documents from the vector store based on the question.

    Args:
        state (GraphState): Current state containing the question

    Returns:
        Dict[str, Any]: Updated state with retrieved documents
    """
    logger.info("Retrieving documents for question")
    print("---RETRIEVE---")

    question = state["question"]

    # Retrieve documents from the vector store
    documents = retriever.invoke(question)
    logger.info(f"Retrieved {len(documents)} documents")

    return {"documents": documents, "question": question}
