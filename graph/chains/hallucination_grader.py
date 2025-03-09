"""
Chain for grading whether an answer is grounded in the provided documents.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

from graph.chains.models import GradeHallucinations

# Initialize the LLM
llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations, method="function_calling")

# Define the system prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

# Create the prompt template
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# Create the grader chain
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
