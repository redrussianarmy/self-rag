"""
Tests for the answer_grader module.
"""
import pytest
from unittest.mock import patch, MagicMock

from graph.chains.models import GradeAnswer
from graph.chains.answer_grader import answer_grader, answer_prompt, structured_llm_grader


class TestAnswerGrader:
    """Test cases for the answer_grader module."""

    def test_answer_grader_structure(self):
        """Test the structure of the answer_grader module."""
        # Assert that the components exist
        assert answer_prompt is not None
        assert structured_llm_grader is not None
        assert answer_grader is not None

        # Check that answer_grader is a RunnableSequence
        assert hasattr(answer_grader, "invoke")

    def test_answer_prompt_structure(self):
        """Test the structure of the answer prompt."""
        # Check that the prompt has the expected structure
        assert hasattr(answer_prompt, "messages")

        # Check that the prompt contains the expected variables
        assert "question" in answer_prompt.input_variables
        assert "generation" in answer_prompt.input_variables

    def test_structured_llm_grader_existence(self):
        """Test that the structured LLM grader exists."""
        # Check that the structured_llm_grader exists
        assert structured_llm_grader is not None

        # Check that it has a string representation
        assert isinstance(str(structured_llm_grader), str)

    def test_grade_answer_model(self):
        """Test the GradeAnswer model."""
        # Create a GradeAnswer instance
        grade = GradeAnswer(binary_score=True)

        # Check that it has the expected structure
        assert hasattr(grade, "binary_score")
        assert grade.binary_score is True

        # Create another instance with different value
        grade_false = GradeAnswer(binary_score=False)
        assert grade_false.binary_score is False
