"""
Tests for the state module.
"""
import pytest

from graph.state import GraphState


class TestState:
    """Test cases for the state module."""

    def test_graph_state_creation(self):
        """Test creating a GraphState instance."""
        # Create a GraphState instance
        state = GraphState(
            question="What is RAG?",
            generation="RAG is retrieval augmented generation.",
            web_search=False,
            documents=["Document about RAG"]
        )

        # Assert that the state has the expected attributes
        assert state["question"] == "What is RAG?"
        assert state["generation"] == "RAG is retrieval augmented generation."
        assert state["web_search"] is False
        assert state["documents"] == ["Document about RAG"]

    def test_graph_state_required_fields(self):
        """Test GraphState with only required fields."""
        # Create a GraphState instance with minimal parameters
        state = GraphState(
            question="What is RAG?",
            generation="",
            web_search=False,
            documents=[]
        )

        # Assert that the state has the expected attributes
        assert state["question"] == "What is RAG?"
        assert state["generation"] == ""
        assert state["web_search"] is False
        assert state["documents"] == []

    def test_graph_state_update(self):
        """Test updating a GraphState instance."""
        # Create a GraphState instance
        state = GraphState(
            question="What is RAG?",
            generation="",
            web_search=False,
            documents=[]
        )

        # Update the state
        state["generation"] = "RAG is retrieval augmented generation."
        state["web_search"] = True
        state["documents"] = ["Document about RAG"]

        # Assert that the state has been updated
        assert state["generation"] == "RAG is retrieval augmented generation."
        assert state["web_search"] is True
        assert state["documents"] == ["Document about RAG"]
