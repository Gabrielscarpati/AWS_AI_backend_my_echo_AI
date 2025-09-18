"""
Orchestrator module that coordinates all chatbot components.
This file replaces the main chatbot_clio.py logic by organizing all modules.
"""
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, START, END

from .state import State
from .retrieval import retrieve_context
from .conversation import generate_influencer_answer
from .security import security_check_node, regenerate_safe_response, should_retry_security
from .summarization import summarize

# Build the state graph that orchestrates all modules
graph_builder = StateGraph(State)

# Add all nodes
graph_builder.add_node(retrieve_context)
graph_builder.add_node(generate_influencer_answer)
graph_builder.add_node(security_check_node)
graph_builder.add_node(regenerate_safe_response)
graph_builder.add_node(summarize)

# Define the flow
graph_builder.add_edge(START, 'retrieve_context')
graph_builder.add_edge('retrieve_context', 'generate_influencer_answer')
graph_builder.add_edge('generate_influencer_answer', 'security_check_node')

# Conditional edge: if security check fails, retry or proceed to summarize
graph_builder.add_conditional_edges(
    'security_check_node',
    should_retry_security,
    {
        "regenerate_safe_response": "regenerate_safe_response",
        "summarize": "summarize"
    }
)

# After regenerating, check security again
graph_builder.add_edge('regenerate_safe_response', 'security_check_node')
graph_builder.add_edge('summarize', END)

# Compile the chatbot graph
chatbot_clio = graph_builder.compile()

# Export the pattern for backward compatibility
from .config import PATTERN_USER
