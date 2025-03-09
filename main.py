from graph.graph import app
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def run_rag_query(question):
    """
    Run a query through the Self-RAG system

    Args:
        question (str): The question to ask

    Returns:
        dict: The response from the RAG system
    """
    print(f"Processing query: {question}")
    return app.invoke(input={"question": question})


if __name__ == "__main__":
    print("=" * 50)
    print("Self-RAG System with Quality Control")
    print("=" * 50)

    # Example query
    example_query = "What is retrieval-augmented generation?"
    print(f"\nRunning example query: '{example_query}'")
    result = run_rag_query(example_query)
    print("\nResult:")
    print(result)

    # Interactive mode
    while True:
        print("\n" + "-" * 50)
        user_query = input("Enter your question (or 'exit' to quit): ")
        if user_query.lower() in ['exit', 'quit', 'q']:
            break

        result = run_rag_query(user_query)
        print("\nResult:")
        print(result)
