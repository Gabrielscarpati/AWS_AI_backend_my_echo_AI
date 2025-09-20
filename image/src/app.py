import json
import requests
import base64
import os

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
            "msgs_cnt_by_user": ...,
            "custom_voice_url": <optional custom voice URL for TTS>
        }
    }
    Returns JSON {
        "response": "...", 
        "response_type": "text" or "audio",
        "summary_generated": bool, 
        "message_summary": str,
        "audio_url": <if response_type is "audio">,
        "audio_duration": <if response_type is "audio">,
        "tts_metadata": <if response_type is "audio">
    }
    """
    
    try:
        raw_body = event.get("body")
        
        if event.get("isBase64Encoded"):
            raw_body = base64.b64decode(raw_body).decode("utf-8")
            
        try:
            payload = json.loads(raw_body)
        except Exception as e:
            print(f"JSON parsing error: {e}")
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
        custom_voice_url = payload.get("custom_voice_url")  # Optional custom voice for TTS
        
        # Convert chat history to LangChain message objects
        # Handle both formats: ["text1", "text2"] ouand [["user", "text1"], ["assistant", "text2"]]
        try:
            chat_history = []
            for item in chat_history_texts:
                if isinstance(item, str):
                    # Simple string format - assume user message
                    chat_history.append(HumanMessage(content=item))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    # Tuple/list format: [role, content]
                    role, content = item[0], item[1]
                    if role.lower() in ['user', 'human']:
                        chat_history.append(HumanMessage(content=content))
                    elif role.lower() in ['assistant', 'ai', 'bot']:
                        chat_history.append(AIMessage(content=content))
                    else:
                        # Default to user message for unknown roles
                        chat_history.append(HumanMessage(content=content))
                else:
                    print(f"Unexpected chat history item format: {item}")
                    # Try to convert to string as fallback
                    chat_history.append(HumanMessage(content=str(item)))
        except Exception as e:
            print(f"Error creating chat history: {e}")
            print(f"Chat history format received: {chat_history_texts}")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Invalid chat history format: {str(e)}"})
            }
        
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
        
        # Check for required environment variables
        required_env_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            print(f"Missing environment variables: {missing_vars}")
            return {
                "statusCode": 500,
                "body": json.dumps({"error": f"Server configuration error: missing {', '.join(missing_vars)}"})
            }
        
        state = {
            "user_id": user_id, 
            "chat_history": chat_history,
            "msgs_cnt_by_user": msgs_cnt_by_user,
            "creator_id": creator_id,
            "influencer_name": influencer_name,
            "influencer_personality_prompt": influencer_personality_prompt,
            "custom_voice_url": custom_voice_url,
        }
        
        import time
        t_start = time.perf_counter()
        
        # Add comprehensive error handling around the orchestrator invocation
        try:
            final_state = chatbot_clio.invoke(state)
        except Exception as e:
            print(f"Orchestrator error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "statusCode": 500,
                "body": json.dumps({"error": f"Internal processing error: {str(e)}"})
            }
            
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

        # Build response with TTS support
        response_data = {
            "response": final_state.get("response", ""),
            "response_type": final_state.get("response_type", "text"),
            "summary_generated": final_state.get("summary_generated", False),
            "message_summary": final_state.get("message_summary", ""),
            "security": security_info,
            "timings": timings,
            "timings_total": timings_total,
            "wall_time": wall_time,
        }

        # Add audio-specific fields if this is an audio response
        if final_state.get("response_type") == "audio":
            response_data.update({
                "audio_url": final_state.get("audio_url"),
                "audio_duration": final_state.get("audio_duration"),
                "tts_metadata": final_state.get("tts_metadata", {})
            })

        return {
            "statusCode": 200,
            "body": json.dumps(response_data)
        }
    
    except Exception as e:
        print(f"Unexpected error in handler: {e}")
        import traceback
        traceback.print_exc()
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error"})
        }