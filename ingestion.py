from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Define URLs for knowledge base
# These URLs contain information about AI, machine learning, and related topics
urls = [
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Prompt_engineering",
]

# Load documents from URLs
print("Loading documents from URLs...")
docs = [WebBaseLoader(url).load() for url in urls]
# Flatten the list of documents
docs_list = [item for sublist in docs for item in sublist]

# Split documents into smaller chunks for better retrieval
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Create and persist the vector store
# Uncomment this section to recreate the vector store
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
    persist_directory="./.chroma",
)

# Create a retriever from the persisted vector store
print("Creating retriever from vector store...")
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()
