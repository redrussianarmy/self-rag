import logging
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("self_rag")


def decide_to_generate(state: Dict[str, Any]) -> str:
    """
    Decide whether to generate an answer or perform web search based on document relevance.

    Args:
        state: Current state of the workflow

    Returns:
        str: Next node to execute (GENERATE or WEBSEARCH)
    """
    logger.info("Assessing graded documents")
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        logger.info("Decision: Not all documents are relevant to question, including web search")
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        logger.info("Decision: Generate answer from documents")
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """
    Grade whether the generated answer is grounded in documents and addresses the question.

    Args:
        state: Current state of the workflow

    Returns:
        str: Result of grading ("useful", "not useful", or "not supported")
    """
    logger.info("Checking for hallucinations")
    print("---CHECK HALLUCINATIONS---")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Check if generation is grounded in documents
    hallucination_score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_score.binary_score:
        logger.info("Generation is grounded in documents")
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")

        # Check if generation addresses the question
        answer_score = answer_grader.invoke({"question": question, "generation": generation})

        if answer_score.binary_score:
            logger.info("Generation addresses question")
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            logger.info("Generation does not address question")
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        logger.info("Generation is not grounded in documents, retrying")
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def create_workflow() -> StateGraph:
    """
    Create and configure the workflow graph.

    Returns:
        StateGraph: Configured workflow graph
    """
    # Create the workflow graph
    logger.info("Creating workflow graph")
    workflow = StateGraph(GraphState)

    # Add nodes to the graph
    workflow.add_node(RETRIEVE, retrieve)
    workflow.add_node(GRADE_DOCUMENTS, grade_documents)
    workflow.add_node(GENERATE, generate)
    workflow.add_node(WEBSEARCH, web_search)

    # Set entry point
    workflow.set_entry_point(RETRIEVE)

    # Add edges between nodes
    workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
    workflow.add_conditional_edges(
        GRADE_DOCUMENTS,
        decide_to_generate,
        {
            WEBSEARCH: WEBSEARCH,
            GENERATE: GENERATE,
        },
    )

    workflow.add_conditional_edges(
        GENERATE,
        grade_generation_grounded_in_documents_and_question,
        {
            "not supported": GENERATE,
            "useful": END,
            "not useful": WEBSEARCH,
        },
    )
    workflow.add_edge(WEBSEARCH, GENERATE)
    workflow.add_edge(GENERATE, END)

    return workflow


# Create and compile the graph
workflow = create_workflow()
app = workflow.compile()

# Generate visualization
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"graph_{timestamp}.png"
logger.info(f"Generating graph visualization: {output_file}")
app.get_graph().draw_mermaid_png(output_file_path=output_file)
