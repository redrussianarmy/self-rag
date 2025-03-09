"""
Tests for the graph module.
"""
import pytest
from unittest.mock import patch, MagicMock

from graph.graph import decide_to_generate, grade_generation_grounded_in_documents_and_question, create_workflow, app
from graph.state import GraphState
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH


class TestGraph:
    """Test cases for the graph module."""

    def test_decide_to_generate_with_web_search(self):
        """Test decide_to_generate when web_search is True."""
        # Setup
        state = {"web_search": True}

        # Execute
        result = decide_to_generate(state)

        # Assert
        assert result == WEBSEARCH

    def test_decide_to_generate_without_web_search(self):
        """Test decide_to_generate when web_search is False."""
        # Setup
        state = {"web_search": False}

        # Execute
        result = decide_to_generate(state)

        # Assert
        assert result == GENERATE

    @patch("graph.graph.hallucination_grader")
    def test_grade_generation_grounded(self, mock_hallucination_grader):
        """Test grade_generation_grounded_in_documents_and_question when generation is grounded."""
        # Setup
        mock_result = MagicMock()
        mock_result.binary_score = True
        mock_hallucination_grader.invoke.return_value = mock_result

        # Setup for answer_grader
        with patch("graph.graph.answer_grader") as mock_answer_grader:
            mock_answer_result = MagicMock()
            mock_answer_result.binary_score = True
            mock_answer_grader.invoke.return_value = mock_answer_result

            # Setup state
            state = GraphState(
                question="What is RAG?",
                generation="RAG is retrieval augmented generation.",
                web_search=False,
                documents=["Document about RAG"]
            )

            # Execute
            result = grade_generation_grounded_in_documents_and_question(state)

            # Assert
            assert result == "useful"
            mock_hallucination_grader.invoke.assert_called_once()
            mock_answer_grader.invoke.assert_called_once()

    @patch("graph.graph.hallucination_grader")
    def test_grade_generation_not_grounded(self, mock_hallucination_grader):
        """Test grade_generation_grounded_in_documents_and_question when generation is not grounded."""
        # Setup
        mock_result = MagicMock()
        mock_result.binary_score = False
        mock_hallucination_grader.invoke.return_value = mock_result

        # Setup state
        state = GraphState(
            question="What is RAG?",
            generation="RAG is retrieval augmented generation.",
            web_search=False,
            documents=["Document about RAG"]
        )

        # Execute
        result = grade_generation_grounded_in_documents_and_question(state)

        # Assert
        assert result == "not supported"
        mock_hallucination_grader.invoke.assert_called_once()

    @patch("graph.graph.hallucination_grader")
    def test_grade_generation_not_addressing_question(self, mock_hallucination_grader):
        """Test grade_generation_grounded_in_documents_and_question when generation doesn't address question."""
        # Setup
        mock_result = MagicMock()
        mock_result.binary_score = True
        mock_hallucination_grader.invoke.return_value = mock_result

        # Setup for answer_grader
        with patch("graph.graph.answer_grader") as mock_answer_grader:
            mock_answer_result = MagicMock()
            mock_answer_result.binary_score = False
            mock_answer_grader.invoke.return_value = mock_answer_result

            # Setup state
            state = GraphState(
                question="What is RAG?",
                generation="Machine learning is a subset of AI.",
                web_search=False,
                documents=["Document about RAG"]
            )

            # Execute
            result = grade_generation_grounded_in_documents_and_question(state)

            # Assert
            assert result == "not useful"
            mock_hallucination_grader.invoke.assert_called_once()
            mock_answer_grader.invoke.assert_called_once()

    def test_create_workflow(self):
        """Test create_workflow function."""
        # Execute
        workflow = create_workflow()

        # Assert
        assert workflow is not None
        assert hasattr(workflow, "add_node")
        assert hasattr(workflow, "add_edge")
        assert hasattr(workflow, "add_conditional_edges")
        assert hasattr(workflow, "compile")

    def test_app_existence(self):
        """Test that the app exists."""
        assert app is not None
        assert hasattr(app, "invoke")
        assert hasattr(app, "get_graph")
