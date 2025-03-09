"""
Tests for the retrieval_grader module.
"""
import pytest
from unittest.mock import patch, MagicMock

from graph.chains.models import GradeDocuments
from graph.chains.retrieval_grader import retrieval_grader, grade_prompt, structured_llm_grader


class TestRetrievalGrader:
    """Test cases for the retrieval_grader module."""

    def test_retrieval_grader_structure(self):
        """Test the structure of the retrieval_grader module."""
        # Assert that the components exist
        assert grade_prompt is not None
        assert structured_llm_grader is not None
        assert retrieval_grader is not None

        # Check that retrieval_grader has invoke method
        assert hasattr(retrieval_grader, "invoke")

    def test_grade_prompt_structure(self):
        """Test the structure of the grade prompt."""
        # Check that the prompt has the expected structure
        assert hasattr(grade_prompt, "messages")

        # Check that the prompt contains the expected variables
        assert "document" in grade_prompt.input_variables
        assert "question" in grade_prompt.input_variables

    def test_structured_llm_grader_existence(self):
        """Test that the structured LLM grader exists."""
        # Check that the structured_llm_grader exists
        assert structured_llm_grader is not None

        # Check that it has a string representation
        assert isinstance(str(structured_llm_grader), str)

    def test_grade_documents_model(self):
        """Test the GradeDocuments model."""
        # Create a GradeDocuments instance
        grade = GradeDocuments(binary_score="yes")

        # Check that it has the expected structure
        assert hasattr(grade, "binary_score")
        assert grade.binary_score == "yes"

        # Create another instance with different value
        grade_no = GradeDocuments(binary_score="no")
        assert grade_no.binary_score == "no"

    def test_grade_prompt_content(self):
        """Test the content of the grade prompt."""
        # Convert the prompt to a string to check its content
        prompt_str = str(grade_prompt)

        # Check that the prompt contains key phrases
        assert "grader" in prompt_str.lower()
        assert "relevance" in prompt_str.lower()
        assert "document" in prompt_str.lower()
        assert "question" in prompt_str.lower()

        # Check that the prompt has placeholders for document and question
        assert "{document}" in prompt_str
        assert "{question}" in prompt_str
