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
from .media_processing import process_media_input, generate_tts_output
import os
from supabase import create_client
import threading
import json

def log_interaction(state: State) -> State:
    """Log the interaction to Supabase in a background thread."""
    def insert_row():
        client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
        
        # Ensure recent_chat_history_json is available (fallback if missing)
        if 'recent_chat_history_json' not in state:
            full_chat_history = state.get('chat_history', [])
            recent_chat_history = full_chat_history[-10:]  # Last 10 messages as fallback
            recent_history_json = json.dumps([{"role": "USER" if msg.type == "human" else "ASSISTANT", "content": msg.content} for msg in recent_chat_history])
            state['recent_chat_history_json'] = recent_history_json
            print("Fallback: Serialized recent_chat_history from raw state")
        
        data = {
            "user_question": state.get('user_query', ''),
            "creator_id": state.get('creator_id', ''),
            "user_id": state.get('user_id', ''),
            "influencer_name": state.get('influencer_name', ''),
            "conversation_summaries": state.get('retrieved_summaries', ''),
            "personality": state.get('influencer_personality_prompt', ''),
            "recent_chat_history": state.get('recent_chat_history_json', json.dumps([])),
            "interview_and_communication_style": state.get('interview_and_communication_style_json', json.dumps([])),
            "expert_analysis": state.get('expert_analysis_json', json.dumps([])),
            "context_data": state.get('context_data_json', json.dumps([])),
            "input_media_type": state.get('input_media_type', 'text'),
            "transcription": (state.get('audio_transcription') or state.get('image_transcription', '') or ''),
            "should_generate_tts": state.get('should_generate_tts', False),
            "answer": state.get('response', '')  # Ensure non-null
        }
        
        # Debug: Print the data being inserted
        print("Logging data to Supabase:", json.dumps(data, indent=2))
        
        try:
            client.table('prompt_history').insert(data).execute()
            print("Interaction logged to Supabase successfully.")
        except Exception as e:
            print(f"Failed to log interaction to Supabase: {e}")

    threading.Thread(target=insert_row, daemon=True).start()
    return state

# Build the state graph that orchestrates all modules
graph_builder = StateGraph(State)

# Add all nodes
graph_builder.add_node(process_media_input)
graph_builder.add_node(retrieve_context)
graph_builder.add_node(generate_influencer_answer)
graph_builder.add_node(security_check_node)
graph_builder.add_node(regenerate_safe_response)
graph_builder.add_node(summarize)
graph_builder.add_node(generate_tts_output)
graph_builder.add_node(log_interaction)

# Define the flow
graph_builder.add_edge(START, 'process_media_input')
graph_builder.add_edge('process_media_input', 'retrieve_context')
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
graph_builder.add_edge('summarize', 'generate_tts_output')
graph_builder.add_edge('generate_tts_output', 'log_interaction')
graph_builder.add_edge('log_interaction', END)

# Compile the chatbot graph
chatbot_clio = graph_builder.compile()

# Export the pattern for backward compatibility
from .config import PATTERN_USER
