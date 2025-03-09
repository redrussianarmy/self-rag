from typing import Any, Dict
import logging

from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from graph.state import GraphState

logger = logging.getLogger("self_rag.web_search")

# Initialize the web search tool with more results
web_search_tool = TavilySearchResults(k=5, include_domains=["wikipedia.org", "en.wikipedia.org"])


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Perform a web search to supplement the retrieved documents.

    Args:
        state (GraphState): The current state of the graph containing the question

    Returns:
        Dict[str, Any]: Updated state with additional documents from web search
    """
    logger.info("Performing web search to supplement retrieved documents")
    print("---WEB SEARCH---")

    question = state["question"]
    documents = state["documents"]

    # Log the search query
    logger.info(f"Searching web for: {question}")

    try:
        # Perform the web search
        search_results = web_search_tool.invoke({"query": question})

        # Log the number of results found
        logger.info(f"Found {len(search_results)} web search results")

        # Format the results as a document
        web_results = "\n\n".join([f"Source: {d['url']}\n{d['content']}" for d in search_results])
        web_results_doc = Document(page_content=web_results, metadata={"source": "web_search"})

        # Add the web search results to the documents
        if documents is not None:
            documents.append(web_results_doc)
            logger.info(f"Added web search results to documents. Total documents: {len(documents)}")
        else:
            documents = [web_results_doc]
            logger.info("Created new documents list with web search results")

        return {"documents": documents, "question": question}

    except Exception as e:
        logger.error(f"Error during web search: {str(e)}")
        # Return the original documents if web search fails
        return {"documents": documents if documents else [], "question": question}
