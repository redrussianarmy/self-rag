# Self-RAG: Retrieval Augmented Generation with Quality Control

This project implements a Self-RAG (Retrieval Augmented Generation) system with built-in quality control mechanisms. The system uses LangGraph to create a workflow that retrieves relevant documents, grades their relevance, generates answers, and checks for hallucinations.

## Features

- **Document Retrieval**: Retrieves relevant documents from a vector database
- **Document Grading**: Evaluates the relevance of retrieved documents to the question
- **Web Search Fallback**: Automatically performs web search when local documents are insufficient
- **Answer Generation**: Generates answers based on retrieved documents
- **Hallucination Detection**: Checks if generated answers are grounded in the retrieved documents
- **Answer Quality Assessment**: Evaluates if the answer addresses the original question
- **Comprehensive Testing**: 100% test coverage with unit tests for all components
- **Robust Logging**: Detailed logging for better debugging and monitoring
- **Error Handling**: Graceful error handling for web search and other components

## Architecture

The system is built using LangGraph, which allows for the creation of complex workflows with conditional branching. The workflow consists of the following nodes:

1. **Retrieve**: Retrieves documents from the vector database
2. **Grade Documents**: Evaluates the relevance of documents to the question
3. **Web Search**: Performs web search if local documents are insufficient
4. **Generate**: Generates an answer based on the documents
5. **Grade Generation**: Checks if the answer is grounded in the documents and addresses the question

### Workflow Diagram

The system automatically generates a workflow diagram when run, showing the connections between different components. The diagram is saved as `graph_[timestamp].png` in the project directory.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/self-rag.git
cd self-rag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key  # For web search
```

## Usage

1. Run the ingestion script to create the vector database (first time only):
```bash
python ingestion.py
```

2. Uncomment the vectorstore creation code in `ingestion.py` if you need to recreate the database.

3. Run the main application:
```bash
python main.py
```

4. Enter your questions when prompted or exit by typing 'exit'.

## Testing

The project includes comprehensive tests for all components:

```bash
# Run all tests
python -m pytest

# Run tests with coverage report
python -m pytest --cov=graph

# Run specific test modules
python -m pytest tests/graph/chains/
python -m pytest tests/graph/nodes/
```

The test suite includes:
- Unit tests for all chain components
- Unit tests for all node components
- Unit tests for graph, state, and constants
- Mock-based tests to avoid external API calls during testing

## Project Structure

```
.
├── .env                  # Environment variables (create this file)
├── README.md             # This file
├── ingestion.py          # Document ingestion script
├── main.py               # Main application entry point
├── requirements.txt      # Project dependencies
├── rag_system.log        # System logs
├── graph/                # LangGraph workflow components
│   ├── __init__.py
│   ├── consts.py         # Constants used in the graph
│   ├── graph.py          # Main graph definition
│   ├── state.py          # State definition for the graph
│   ├── chains/           # LangChain chains used in the graph
│   │   ├── __init__.py
│   │   ├── answer_grader.py
│   │   ├── generation.py
│   │   ├── hallucination_grader.py
│   │   ├── models.py     # Shared Pydantic models
│   │   └── retrieval_grader.py
│   └── nodes/            # Graph nodes implementation
│       ├── __init__.py
│       ├── generate.py
│       ├── grade_documents.py
│       ├── retrieve.py
│       └── web_search.py
├── tests/                # Test suite
│   ├── __init__.py
│   └── graph/
│       ├── __init__.py
│       ├── test_consts.py
│       ├── test_graph.py
│       ├── test_state.py
│       ├── chains/
│       │   ├── __init__.py
│       │   ├── test_answer_grader.py
│       │   ├── test_generation.py
│       │   ├── test_hallucination_grader.py
│       │   └── test_retrieval_grader.py
│       └── nodes/
│           ├── __init__.py
│           ├── test_generate.py
│           ├── test_grade_documents.py
│           ├── test_retrieve.py
│           └── test_web_search.py
└── .chroma/              # Vector database (created by ingestion.py)
```

## Customization

- **Knowledge Sources**: Modify the URLs in `ingestion.py` to use different knowledge sources
- **Document Chunking**: Adjust the chunk size in `ingestion.py` to change how documents are split
- **Prompts**: Modify the prompts in the chain files to customize the behavior of the system
- **Web Search**: Configure the web search parameters in `graph/nodes/web_search.py`
- **Logging**: Adjust logging levels and handlers in `graph/graph.py`

## Performance Optimization

- The system uses caching to avoid redundant API calls
- Document chunking is optimized for retrieval performance
- Web search is only triggered when necessary
- Error handling ensures the system continues to function even when components fail

## License

MIT

## Acknowledgements

This project is built using:
- [LangChain](https://github.com/langchain-ai/langchain) - Framework for building LLM applications
- [LangGraph](https://github.com/langchain-ai/langgraph) - Framework for building stateful, multi-actor applications
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database for document storage
- [OpenAI](https://openai.com/) - LLM provider for text generation and embeddings
- [Tavily](https://tavily.com/) - API for web search capabilities
- [pytest](https://docs.pytest.org/) - Testing framework