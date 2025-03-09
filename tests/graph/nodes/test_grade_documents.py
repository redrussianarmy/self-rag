"""
Tests for the grade_documents node.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import Document

from graph.nodes.grade_documents import grade_documents
from graph.state import GraphState


class TestGradeDocumentsNode:
    """Test cases for the grade_documents node."""

    @patch("graph.nodes.grade_documents.retrieval_grader")
    def test_grade_documents_structure(self, mock_grader):
        """Test the structure of the grade_documents function."""
        # Check that grade_documents is a function
        assert callable(grade_documents)

    @patch("graph.nodes.grade_documents.retrieval_grader")
    def test_grade_documents_all_relevant(self, mock_grader):
        """Test grade_documents when all documents are relevant."""
        # Setup
        mock_result = MagicMock()
        mock_result.binary_score = "yes"
        mock_grader.invoke.return_value = mock_result

        question = "What is RAG?"
        doc1 = Document(page_content="RAG is retrieval augmented generation.")
        doc2 = Document(page_content="RAG combines retrieval with generation.")
        state = GraphState(question=question, generation="", web_search=False, documents=[doc1, doc2])

        # Execute
        result = grade_documents(state)

        # Assert
        assert "documents" in result
        assert "question" in result
        assert "web_search" in result
        assert result["question"] == question
        assert len(result["documents"]) == 2
        assert result["web_search"] is False
        assert mock_grader.invoke.call_count == 2

    @patch("graph.nodes.grade_documents.retrieval_grader")
    def test_grade_documents_some_irrelevant(self, mock_grader):
        """Test grade_documents when some documents are irrelevant."""
        # Setup
        def mock_invoke(inputs):
            # Return "yes" for the first document, "no" for the second
            if "RAG is retrieval" in inputs["document"]:
                result = MagicMock()
                result.binary_score = "yes"
                return result
            else:
                result = MagicMock()
                result.binary_score = "no"
                return result

        mock_grader.invoke.side_effect = mock_invoke

        question = "What is RAG?"
        doc1 = Document(page_content="RAG is retrieval augmented generation.")
        doc2 = Document(page_content="Machine learning is a subset of AI.")
        state = GraphState(question=question, generation="", web_search=False, documents=[doc1, doc2])

        # Execute
        result = grade_documents(state)

        # Assert
        assert "documents" in result
        assert "question" in result
        assert "web_search" in result
        assert result["question"] == question
        assert len(result["documents"]) == 1
        assert result["documents"][0].page_content == "RAG is retrieval augmented generation."
        assert result["web_search"] is True

    @patch("graph.nodes.grade_documents.retrieval_grader")
    def test_grade_documents_all_irrelevant(self, mock_grader):
        """Test grade_documents when all documents are irrelevant."""
        # Setup
        mock_result = MagicMock()
        mock_result.binary_score = "no"
        mock_grader.invoke.return_value = mock_result

        question = "What is RAG?"
        doc1 = Document(page_content="Machine learning is a subset of AI.")
        doc2 = Document(page_content="Deep learning uses neural networks.")
        state = GraphState(question=question, generation="", web_search=False, documents=[doc1, doc2])

        # Execute
        result = grade_documents(state)

        # Assert
        assert "documents" in result
        assert "question" in result
        assert "web_search" in result
        assert result["question"] == question
        assert len(result["documents"]) == 0
        assert result["web_search"] is True
        assert mock_grader.invoke.call_count == 2
