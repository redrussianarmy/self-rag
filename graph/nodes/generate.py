from typing import Any, Dict
import logging

from graph.chains.generation import generation_chain
from graph.state import GraphState

logger = logging.getLogger("self_rag.generate")


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate an answer based on the retrieved documents and question.

    Args:
        state (GraphState): The current state of the graph containing documents and question

    Returns:
        Dict[str, Any]: Updated state with the generated answer
    """
    logger.info("Generating answer from retrieved documents")
    print("---GENERATE---")

    question = state["question"]
    documents = state["documents"]

    # Log the number of documents being used for generation
    doc_count = len(documents) if documents else 0
    logger.info(f"Using {doc_count} documents for generation")

    # Generate the answer
    generation = generation_chain.invoke({"context": documents, "question": question})

    # Log a preview of the generated answer
    preview = generation[:100] + "..." if len(generation) > 100 else generation
    logger.info(f"Generated answer preview: {preview}")

    return {"documents": documents, "question": question, "generation": generation}
