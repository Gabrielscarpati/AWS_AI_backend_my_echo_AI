import json
import requests
import base64

from chatbot.orchestrator import chatbot_clio, PATTERN_USER
from dotenv import load_dotenv
from supabase_utils import create_message, get_test_credentials
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


def handler(event, context):
    """
    Expects event = {
        "isBase64Encoded": True/False (this one can be omitted)
        "body": {
            "user_id": <user_id>,
            "creator_id": <creator_id>,
            "influencer_name": <optional influencer_name>,
            "influencer_personality_prompt": <optional personality prompt>,
            "chat_history": [<plain_text_message>, ...],  # Changed to plain text
            "msgs_cnt_by_user": ...
        }
    }
    Returns JSON {"response": "...", "summary_generated": bool, "message_summary": str}
    """
    
    raw_body = event.get("body")
    
    if event.get("isBase64Encoded"):
        raw_body = base64.b64decode(raw_body).decode("utf-8")
        
    try:
        payload = json.loads(raw_body)
    except:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid JSON"})
        }
    
    user_id = payload.get("user_id")
    creator_id = payload.get("creator_id")
    influencer_name = payload.get("influencer_name")  # Optional influencer name from frontend
    influencer_personality_prompt = payload.get("influencer_personality_prompt")  # Optional personality prompt
    chat_history_texts = payload.get("chat_history", [])  # List of plain text messages
    msgs_cnt_by_user = payload.get("msgs_cnt_by_user")
    
    # Convert plain text messages to LangChain message objects
    # Assume all messages are from user for simplicity
    chat_history = [HumanMessage(content=text) for text in chat_history_texts]
    
    # Coerce msgs_cnt_by_user to int for robust modulo-based summary logic
    try:
        msgs_cnt_by_user = int(msgs_cnt_by_user)
    except Exception:
        msgs_cnt_by_user = 0
    
    if not user_id or not chat_history:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "user_id and chat_history required"})
        }
    
    state = {
        "user_id": user_id, 
        "chat_history": chat_history,
        "msgs_cnt_by_user": msgs_cnt_by_user,
        "creator_id": creator_id,
        "influencer_name": influencer_name,
        "influencer_personality_prompt": influencer_personality_prompt,
    }
    
    import time
    t_start = time.perf_counter()
    final_state = chatbot_clio.invoke(state)
    wall_time = time.perf_counter() - t_start
    # Some execution paths populate timings into the module-level TIMINGS
    # while others attach it to the returned state. Prefer the returned
    # timings but fall back to the module TIMINGS for visibility.
    timings = final_state.get('timings', {})
    try:
        if not timings:
            from chatbot.retrieval import TIMINGS
            timings = TIMINGS or {}
    except Exception:
        timings = timings or {}

    # Compute total of numeric timing values (ignore non-numeric values)
    try:
        timings_total = sum(v for v in timings.values() if isinstance(v, (int, float)))
    except Exception:
        timings_total = 0.0

    # Include security information in response
    security_info = {
        "security_check_passed": final_state.get("security_check_passed", True),
        "security_flags": final_state.get("security_flags", []),
        "security_retry_count": final_state.get("security_retry_count", 0),
        "security_check_result": final_state.get("security_check_result", {})
    }

    return {
        "statusCode": 200,
        "body": json.dumps({
            "response": final_state.get("response", ""),
            "summary_generated": final_state.get("summary_generated", False),
            "message_summary": final_state.get("message_summary", ""),
            "security": security_info,
            "timings": timings,
            "timings_total": timings_total,
            "wall_time": wall_time,
        })
    }