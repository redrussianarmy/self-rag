"""
Tests for the consts module.
"""
import pytest

from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH


class TestConsts:
    """Test cases for the consts module."""

    def test_consts_values(self):
        """Test that the constants have the expected values."""
        assert RETRIEVE == "retrieve"
        assert GRADE_DOCUMENTS == "grade_documents"
        assert GENERATE == "generate"
        assert WEBSEARCH == "websearch"

    def test_consts_types(self):
        """Test that the constants have the expected types."""
        assert isinstance(RETRIEVE, str)
        assert isinstance(GRADE_DOCUMENTS, str)
        assert isinstance(GENERATE, str)
        assert isinstance(WEBSEARCH, str)

    def test_consts_uniqueness(self):
        """Test that the constants are unique."""
        consts = [RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH]
        assert len(consts) == len(set(consts))
