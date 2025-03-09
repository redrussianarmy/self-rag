"""
Tests for the hallucination_grader module.
"""
import pytest
from unittest.mock import patch, MagicMock

from graph.chains.models import GradeHallucinations
from graph.chains.hallucination_grader import hallucination_grader, hallucination_prompt, structured_llm_grader


class TestHallucinationGrader:
    """Test cases for the hallucination_grader module."""

    def test_hallucination_grader_structure(self):
        """Test the structure of the hallucination_grader module."""
        # Assert that the components exist
        assert hallucination_prompt is not None
        assert structured_llm_grader is not None
        assert hallucination_grader is not None

        # Check that hallucination_grader is a RunnableSequence
        assert hasattr(hallucination_grader, "invoke")

    def test_hallucination_prompt_structure(self):
        """Test the structure of the hallucination prompt."""
        # Check that the prompt has the expected structure
        assert hasattr(hallucination_prompt, "messages")

        # Check that the prompt contains the expected variables
        assert "documents" in hallucination_prompt.input_variables
        assert "generation" in hallucination_prompt.input_variables

    def test_structured_llm_grader_existence(self):
        """Test that the structured LLM grader exists."""
        # Check that the structured_llm_grader exists
        assert structured_llm_grader is not None

        # Check that it has a string representation
        assert isinstance(str(structured_llm_grader), str)

    def test_grade_hallucinations_model(self):
        """Test the GradeHallucinations model."""
        # Create a GradeHallucinations instance
        grade = GradeHallucinations(binary_score=True)

        # Check that it has the expected structure
        assert hasattr(grade, "binary_score")
        assert grade.binary_score is True

        # Create another instance with different value
        grade_false = GradeHallucinations(binary_score=False)
        assert grade_false.binary_score is False

    def test_hallucination_prompt_content(self):
        """Test the content of the hallucination prompt."""
        # Convert the prompt to a string to check its content
        prompt_str = str(hallucination_prompt)

        # Check that the prompt contains key phrases
        assert "grader" in prompt_str.lower()
        assert "grounded" in prompt_str.lower()
        assert "facts" in prompt_str.lower()

        # Check that the prompt has placeholders for documents and generation
        assert "{documents}" in prompt_str
        assert "{generation}" in prompt_str
