"""
Tests for the web_search node.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import Document

from graph.nodes.web_search import web_search, web_search_tool
from graph.state import GraphState


class TestWebSearchNode:
    """Test cases for the web_search node."""

    def test_web_search_structure(self):
        """Test the structure of the web_search function."""
        # Check that web_search is a function
        assert callable(web_search)

        # Check that web_search_tool exists
        assert web_search_tool is not None

    @patch("graph.nodes.web_search.web_search_tool")
    def test_web_search_with_existing_documents(self, mock_tool):
        """Test web_search with existing documents."""
        # Setup
        mock_tool.invoke.return_value = [
            {"content": "RAG is a technique in AI.", "url": "https://example.com/1"},
            {"content": "RAG combines retrieval with generation.", "url": "https://example.com/2"}
        ]

        question = "What is RAG?"
        doc1 = Document(page_content="RAG is retrieval augmented generation.")
        state = GraphState(question=question, generation="", web_search=True, documents=[doc1])

        # Execute
        result = web_search(state)

        # Assert
        assert "documents" in result
        assert "question" in result
        assert result["question"] == question
        assert len(result["documents"]) == 2  # Original document + web search results
        assert result["documents"][0].page_content == "RAG is retrieval augmented generation."
        assert "RAG is a technique in AI." in result["documents"][1].page_content
        assert "RAG combines retrieval with generation." in result["documents"][1].page_content
        mock_tool.invoke.assert_called_once_with({"query": question})

    @patch("graph.nodes.web_search.web_search_tool")
    def test_web_search_with_no_documents(self, mock_tool):
        """Test web_search with no existing documents."""
        # Setup
        mock_tool.invoke.return_value = [
            {"content": "RAG is a technique in AI.", "url": "https://example.com/1"},
            {"content": "RAG combines retrieval with generation.", "url": "https://example.com/2"}
        ]

        question = "What is RAG?"
        state = GraphState(question=question, generation="", web_search=True, documents=None)

        # Execute
        result = web_search(state)

        # Assert
        assert "documents" in result
        assert "question" in result
        assert result["question"] == question
        assert len(result["documents"]) == 1  # Only web search results
        assert "RAG is a technique in AI." in result["documents"][0].page_content
        assert "RAG combines retrieval with generation." in result["documents"][0].page_content
        mock_tool.invoke.assert_called_once_with({"query": question})

    @patch("graph.nodes.web_search.web_search_tool")
    def test_web_search_error_handling(self, mock_tool):
        """Test web_search error handling."""
        # Setup
        mock_tool.invoke.side_effect = Exception("API error")

        question = "What is RAG?"
        doc1 = Document(page_content="RAG is retrieval augmented generation.")
        state = GraphState(question=question, generation="", web_search=True, documents=[doc1])

        # Execute
        result = web_search(state)

        # Assert
        assert "documents" in result
        assert "question" in result
        assert result["question"] == question
        assert len(result["documents"]) == 1  # Original document preserved
        assert result["documents"][0].page_content == "RAG is retrieval augmented generation."
        mock_tool.invoke.assert_called_once_with({"query": question})
