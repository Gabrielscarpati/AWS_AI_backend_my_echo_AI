#!/usr/bin/env python3
"""
Local test script to run the chatbot without Docker.
This simulates the Lambda handler locally.
"""

import sys
import os
import json
from pathlib import Path

# Add the image/src directory to Python path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent / "image" / "src"))

# Change to the image/src directory so relative imports work
os.chdir(Path(__file__).parent / "image" / "src")

from app import handler
from supabase_utils import get_test_credentials

def create_test_event(
    user_message: str,
    user_id: str = None,
    creator_id: str = None,
    influencer_name: str = None,
    influencer_personality_prompt: str = None,
):
    """
    Create a test event similar to what Lambda would receive.
    """
    if not user_id:
        # Try to get test credentials if no user_id provided
        try:
            user_id, _ = get_test_credentials()
        except Exception as e:
            print(f"Warning: Could not get test credentials: {e}")
            user_id = "test-user-123"
    
    # Defaults for local simulation
    if not creator_id:
        creator_id = "taylor_swift"
    if influencer_name is None:
        influencer_name = creator_id
    if influencer_personality_prompt is None:
        influencer_personality_prompt = (
            "Persona: You are Taylor Swift. Speak with warmth, candor, and lyrical wit. "
            "Be thoughtful, authentic, and supportive. Offer specific, practical guidance."
        )

    # Simple chat history with user message
    chat_history = [
        ("user", user_message)
    ]
    
    event = {
        "body": json.dumps({
            "user_id": user_id,
            "creator_id": creator_id,
            "influencer_name": influencer_name,
            "influencer_personality_prompt": influencer_personality_prompt,
            "chat_history": chat_history,
            "msgs_cnt_by_user": 1
        }),
        "isBase64Encoded": False
    }
    
    return event

def main():
    print("🤖 Starting local chatbot test...")
    print("=" * 50)
    
    # Check if environment variables are set
    required_env_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please update your .env file with the correct values.")
        return
    
    print("✅ Environment variables loaded")
    
    # Interactive mode
    print("\nEntering interactive chat mode. Type 'quit' to exit.")
    print("-" * 50)
    
    user_id = None
    creator_id = "taylor_swift"
    influencer_name = creator_id
    influencer_personality_prompt = (
        '''
STYLE GUARDRAILS (Taylor Swift; warm, grounded—not caricature)

COLLOQUIALISM POLICY
- Allowed slang pool (from your original): like, you know, I mean, kinda, honestly, literally, magical, amazing, ridiculous(ly), best-thing-ever, adorable, “when I was 15…”, “I’m such a cat-person”, “oh my gosh”, haha, lol.
- Per-reply budget: ≤1 slang item. Never place slang in the first sentence.
- Cooldown: do not repeat the same slang within the next 5 assistant replies.
- Mirror rule: only use stronger interjections (“oh my gosh”, haha, lol) if the user’s last message is celebratory/casual (exclamation/emoji/slang). Otherwise prefer mild forms (e.g., “kinda”, “totally”).
- No stacked questions; at most one specific question.

NAME USAGE
- Use the user’s name in the first greeting or when clarifying. Never in two consecutive replies; not more than once every 5 turns.

LENGTH
- Default 1–2 concise sentences (≈18–40 words). The cap is a ceiling, not a target. If steps are requested, use ≤5 short bullets.

BEFORE SENDING (self-check)
1) If >1 slang appears, keep the first and remove the rest.
2) If slang appears in sentence 1, move it to sentence 2+.
3) If the same slang was used in the last 5 assistant turns, swap or remove.
4) ≤1 question; answer first, then optional brief follow-up.
        '''
    )
    
    # Try to get test user
    try:
        user_id, access_token = get_test_credentials()
        print(f"✅ Using test user: {user_id[:8]}...")
    except Exception as e:
        print(f"⚠️  Could not get test credentials: {e}")
        user_id = "test-user-local2"

    msg_count = 0
    chat_history = []  # accumulate (role, content) tuples across turns
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            msg_count += 1
            
            # Append user's message to the running chat history
            chat_history.append(("user", user_input))

            # Create test event and override with accumulated history
            event = create_test_event(
                user_input,
                user_id,
                creator_id,
                influencer_name,
                influencer_personality_prompt,
            )
            event["body"] = json.loads(event["body"])  # to dict for mutation
            event["body"]["msgs_cnt_by_user"] = msg_count
            event["body"]["chat_history"] = chat_history
            event["body"] = json.dumps(event["body"])  # back to json string
            
            print("🤔 Haven is thinking...")
            
            # Call the handler
            response = handler(event, {})
            
            if response["statusCode"] == 200:
                response_data = json.loads(response["body"])
                print(f"🤖 Haven: {response_data['response']}")
                # Append assistant message to the running chat history
                chat_history.append(("assistant", response_data.get('response', '')))
                timings = response_data.get('timings', {})
                if timings:
                    # Calculate total avoiding double-counting nested timings
                    # generate_influencer_answer includes influencer_retrieve, format_pack, and answer_with_rag_model_call
                    # So we only count the top-level operations: retrieve_context + generate_influencer_answer
                    top_level_timings = ['retrieve_context', 'generate_influencer_answer']
                    total = sum(timings.get(k, 0) for k in top_level_timings if k in timings)
                    
                    # Print individual timings and corrected total
                    for k, v in timings.items():
                        print(f"   ⏱ {k}: {v:.3f}s")
                    print(f"   ⏱ total: {total:.3f}s")
            else:
                error_data = json.loads(response["body"])
                print(f"❌ Error: {error_data.get('error', 'Unknown error')}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
