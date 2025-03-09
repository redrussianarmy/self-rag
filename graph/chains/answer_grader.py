"""
Chain for grading whether an answer addresses a question.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

from graph.chains.models import GradeAnswer

# Initialize the LLM
llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer, method="function_calling")

# Define the system prompt
system = """You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

# Create the prompt template
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

# Create the grader chain
answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
