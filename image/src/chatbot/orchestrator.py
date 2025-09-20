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
from .tts_service import should_use_tts, generate_tts_response

# TTS generation node
def generate_tts_node(state: State) -> State:
    """Generate TTS audio if randomly selected and conditions are met."""
    # Check if we should generate TTS for this response
    if not should_use_tts():
        return {"response_type": "text"}

    response_text = state.get("response", "")
    if not response_text:
        return {"response_type": "text"}

    # Get custom voice URL from state
    custom_voice = state.get("custom_voice_url")

    try:
        tts_result = generate_tts_response(response_text, custom_voice)

        if tts_result.get("success"):
            return {
                "response_type": "audio",
                "audio_url": tts_result.get("audio_url"),
                "audio_duration": tts_result.get("duration"),
                "tts_metadata": {
                    "generation_time": tts_result.get("generation_time"),
                    "model_used": tts_result.get("model_used"),
                    "custom_voice_used": tts_result.get("custom_voice_used"),
                    "text_length": tts_result.get("text_length")
                }
            }
        else:
            print(f"TTS generation failed: {tts_result.get('error')}")
            return {"response_type": "text"}

    except Exception as e:
        print(f"TTS node error: {str(e)}")
        return {"response_type": "text"}


# Build the state graph that orchestrates all modules
graph_builder = StateGraph(State)

# Add all nodes
graph_builder.add_node(retrieve_context)
graph_builder.add_node(generate_influencer_answer)
graph_builder.add_node(security_check_node)
graph_builder.add_node(regenerate_safe_response)
graph_builder.add_node(generate_tts_node)
graph_builder.add_node(summarize)

# Define the flow
graph_builder.add_edge(START, 'retrieve_context')
graph_builder.add_edge('retrieve_context', 'generate_influencer_answer')
graph_builder.add_edge('generate_influencer_answer', 'security_check_node')

# Conditional edge: if security check fails, retry or proceed to TTS generation
graph_builder.add_conditional_edges(
    'security_check_node',
    should_retry_security,
    {
        "regenerate_safe_response": "regenerate_safe_response",
        "generate_tts_node": "generate_tts_node"
    }
)

# After regenerating, check security again
graph_builder.add_edge('regenerate_safe_response', 'security_check_node')
# After TTS generation (or decision not to use TTS), proceed to summarization
graph_builder.add_edge('generate_tts_node', 'summarize')
graph_builder.add_edge('summarize', END)

# Compile the chatbot graph
chatbot_clio = graph_builder.compile()

# Export the pattern for backward compatibility
from .config import PATTERN_USER
