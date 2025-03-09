"""
Shared Pydantic models for the RAG system.
"""
from pydantic import BaseModel, Field


class GradeAnswer(BaseModel):
    """Model for grading whether an answer addresses a question."""

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Model for grading whether an answer is grounded in facts."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeDocuments(BaseModel):
    """Model for grading whether documents are relevant to a question."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
