import json
import requests
import base64

from chatbot_clio import chatbot_clio
from dotenv import load_dotenv
from supabase_utils import create_message, get_test_credentials
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

from chatbot_clio import PATTERN_USER


def convert_to_langchain_obj(message):
    """
    message: (<role>, <content>)
    """
    base_message_class = {
        'user': HumanMessage,
        'assistant': AIMessage
    }
    
    role, content = message
    return base_message_class[role](content)


def handler(event, context):
    """
    Expects event = {
        "isBase64Encoded": True/False (this one can be omitted)
        "body": {
            "user_id": <user_id>,
            "creator_id": <creator_id>,
            "influencer_name": <optional influencer_name>,
            "influencer_personality_prompt": <optional personality prompt>,
            "chat_history": [(<is_user>, <content>), ...],
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
    chat_history = payload.get("chat_history")
    msgs_cnt_by_user = payload.get("msgs_cnt_by_user")
    # Coerce msgs_cnt_by_user to int for robust modulo-based summary logic
    try:
        msgs_cnt_by_user = int(msgs_cnt_by_user)
    except Exception:
        msgs_cnt_by_user = 0
    
    try:
        chat_history = [convert_to_langchain_obj(chat_obj) for chat_obj in chat_history]
    except:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "chat_history not properly formatted"})
        }
    
    if not user_id or not chat_history:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "user_id, chat_history and msgs_cnt_by_user required"})
        }
    
    state = {
        "user_id": user_id, 
        "chat_history": chat_history,
        "msgs_cnt_by_user": msgs_cnt_by_user,
        "creator_id": creator_id,
        "influencer_name": influencer_name,  # Pass influencer name to the state
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
            import chatbot_clio as cc
            timings = getattr(cc, 'TIMINGS', {}) or {}
    except Exception:
        timings = timings or {}

    # Compute total of numeric timing values (ignore non-numeric values)
    try:
        timings_total = sum(v for v in timings.values() if isinstance(v, (int, float)))
    except Exception:
        timings_total = 0.0

    return {
        "statusCode": 200,
        "body": json.dumps({
            "response": final_state.get("response", ""),
            "summary_generated": final_state.get("summary_generated", False),
            "message_summary": final_state.get("message_summary", ""),
            "timings": timings,
            "timings_total": timings_total,
            "wall_time": wall_time,
        })
    }