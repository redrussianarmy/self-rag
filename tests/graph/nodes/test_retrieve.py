"""
Tests for the retrieve node.
"""
import pytest
from unittest.mock import patch, MagicMock

from graph.nodes.retrieve import retrieve
from graph.state import GraphState


class TestRetrieveNode:
    """Test cases for the retrieve node."""

    @patch("graph.nodes.retrieve.retriever")
    def test_retrieve_structure(self, mock_retriever):
        """Test the structure of the retrieve function."""
        # Check that retrieve is a function
        assert callable(retrieve)

    @patch("graph.nodes.retrieve.retriever")
    def test_retrieve_with_question(self, mock_retriever):
        """Test retrieve with a question."""
        # Setup
        mock_retriever.invoke.return_value = ["doc1", "doc2"]
        question = "What is RAG?"
        state = GraphState(question=question, generation="", web_search=False, documents=[])

        # Execute
        result = retrieve(state)

        # Assert
        assert "documents" in result
        assert "question" in result
        assert result["question"] == question
        assert result["documents"] == ["doc1", "doc2"]
        mock_retriever.invoke.assert_called_once_with(question)

    @patch("graph.nodes.retrieve.retriever")
    def test_retrieve_with_empty_question(self, mock_retriever):
        """Test retrieve with an empty question."""
        # Setup
        mock_retriever.invoke.return_value = []
        question = ""
        state = GraphState(question=question, generation="", web_search=False, documents=[])

        # Execute
        result = retrieve(state)

        # Assert
        assert "documents" in result
        assert "question" in result
        assert result["question"] == ""
        assert result["documents"] == []
        mock_retriever.invoke.assert_called_once_with("")
