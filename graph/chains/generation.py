"""
Chain for generating answers based on retrieved documents.
"""
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Initialize the LLM with temperature 0 for consistent outputs
llm = ChatOpenAI(temperature=0)

# Pull the RAG prompt from LangChain Hub
# This prompt is designed for retrieval-augmented generation
prompt = hub.pull("rlm/rag-prompt")

# Create the generation chain
# This chain takes context (documents) and a question, and generates an answer
generation_chain = prompt | llm | StrOutputParser()
