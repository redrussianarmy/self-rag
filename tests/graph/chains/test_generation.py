"""
Tests for the generation module.
"""
import pytest
from unittest.mock import patch, MagicMock

from graph.chains.generation import generation_chain, llm, prompt


class TestGeneration:
    """Test cases for the generation module."""

    def test_generation_chain_structure(self):
        """Test the structure of the generation chain."""
        # Assert that the components exist
        assert prompt is not None
        assert llm is not None
        assert generation_chain is not None

        # Check that generation_chain has invoke method
        assert hasattr(generation_chain, "invoke")

    def test_llm_structure(self):
        """Test the structure of the LLM."""
        # Check that the LLM has the expected structure
        assert hasattr(llm, "invoke")

        # Check that the LLM has temperature set to 0
        assert llm.temperature == 0

    def test_prompt_existence(self):
        """Test that the prompt exists."""
        # Check that the prompt exists
        assert prompt is not None

        # Check that it has a string representation
        assert isinstance(str(prompt), str)

    def test_generation_chain_composition(self):
        """Test the composition of the generation chain."""
        # Check that the generation chain is a composition of components
        chain_str = str(generation_chain)

        # The chain should contain references to components
        assert "first" in chain_str
        assert "last" in chain_str
        assert "StrOutputParser" in chain_str

        # Check that the chain has the expected structure
        assert hasattr(generation_chain, "invoke")
