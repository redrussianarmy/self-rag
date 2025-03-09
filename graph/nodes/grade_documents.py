"""
Node for grading the relevance of retrieved documents to the question.
"""
import logging
from typing import Any, Dict, List

from langchain.schema import Document
from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

logger = logging.getLogger("self_rag.grade_documents")


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.
    If any document is not relevant, we will set a flag to run web search.

    Args:
        state (GraphState): The current graph state containing documents and question

    Returns:
        Dict[str, Any]: Updated state with filtered documents and web_search flag
    """
    logger.info("Checking document relevance to question")
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    question = state["question"]
    documents = state["documents"]

    filtered_docs: List[Document] = []
    web_search = False

    # Grade each document for relevance
    for doc_index, doc in enumerate(documents):
        logger.info(f"Grading document {doc_index+1}/{len(documents)}")

        # Invoke the retrieval grader to check relevance
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score.binary_score

        # Process based on relevance grade
        if grade.lower() == "yes":
            logger.info(f"Document {doc_index+1} is relevant")
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            logger.info(f"Document {doc_index+1} is not relevant")
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True

    # Log the results
    logger.info(f"Kept {len(filtered_docs)}/{len(documents)} documents")
    logger.info(f"Web search needed: {web_search}")

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search
    }
