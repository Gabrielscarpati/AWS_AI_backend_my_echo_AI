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
            "chat_history": [(<is_user>, <content>), ...],
            "msgs_cnt_by_user": ...
        }
    }
    Returns JSON {"response": "..."}
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
    chat_history = payload.get("chat_history")
    msgs_cnt_by_user = payload.get("msgs_cnt_by_user")
    
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
    }
    final_state = chatbot_clio.invoke(state)
    
    return {
        "statusCode": 200,
        "body": json.dumps({"response": final_state["response"]})
    }