"""
Tests for the generate node.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import Document

from graph.nodes.generate import generate
from graph.state import GraphState


class TestGenerateNode:
    """Test cases for the generate node."""

    @patch("graph.nodes.generate.generation_chain")
    def test_generate_structure(self, mock_chain):
        """Test the structure of the generate function."""
        # Check that generate is a function
        assert callable(generate)

    @patch("graph.nodes.generate.generation_chain")
    def test_generate_with_documents(self, mock_chain):
        """Test generate with documents."""
        # Setup
        mock_chain.invoke.return_value = "RAG is a technique that combines retrieval with generation."

        question = "What is RAG?"
        doc1 = Document(page_content="RAG is retrieval augmented generation.")
        doc2 = Document(page_content="RAG combines retrieval with generation.")
        state = GraphState(question=question, generation="", web_search=False, documents=[doc1, doc2])

        # Execute
        result = generate(state)

        # Assert
        assert "documents" in result
        assert "question" in result
        assert "generation" in result
        assert result["question"] == question
        assert result["documents"] == [doc1, doc2]
        assert result["generation"] == "RAG is a technique that combines retrieval with generation."
        mock_chain.invoke.assert_called_once()

        # Check that the context and question were passed to the chain
        call_args = mock_chain.invoke.call_args[0][0]
        assert "context" in call_args
        assert "question" in call_args
        assert call_args["question"] == question
        assert call_args["context"] == [doc1, doc2]

    @patch("graph.nodes.generate.generation_chain")
    def test_generate_with_empty_documents(self, mock_chain):
        """Test generate with empty documents."""
        # Setup
        mock_chain.invoke.return_value = "I don't have enough information to answer that question."

        question = "What is RAG?"
        state = GraphState(question=question, generation="", web_search=False, documents=[])

        # Execute
        result = generate(state)

        # Assert
        assert "documents" in result
        assert "question" in result
        assert "generation" in result
        assert result["question"] == question
        assert result["documents"] == []
        assert result["generation"] == "I don't have enough information to answer that question."
        mock_chain.invoke.assert_called_once()

        # Check that the context and question were passed to the chain
        call_args = mock_chain.invoke.call_args[0][0]
        assert "context" in call_args
        assert "question" in call_args
        assert call_args["question"] == question
        assert call_args["context"] == []
