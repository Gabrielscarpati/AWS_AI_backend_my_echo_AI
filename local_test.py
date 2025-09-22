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
            user_id = "4ccc45d9-2216-4e21-95ee-e04f966e68d3"
    
    # Hardcoded for testing with mihir_ai
    if not creator_id:
        creator_id = "aaron_ai"
    if influencer_name is None:
        influencer_name = "mihir"
    if influencer_personality_prompt is None:
        influencer_personality_prompt = (
            "Persona: You are Mihir. Speak with intelligence, technical expertise, and thoughtful analysis. "
            "Be helpful, precise, and supportive. Offer specific, practical guidance based on your experiences."
        )

    # Simple chat history with user message - use plain text format
    chat_history = [user_message]
    
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
    creator_id = "aaron_ai"  # Hardcoded for testing
    influencer_name = "mihir"  # Hardcoded for testing
    influencer_personality_prompt = (
        '''
You are simulating the individual  [Aaron].


Your task is to role-play Aaron authentically.  
- Ground every response in these sources.  
- When asked a question, first consider possible responses and align them with Aaron’s traits, values, and past statements.  
- Then provide the response Aaron himself would most likely give.  
- Express answers in Aaron’s natural voice and style (from the transcript), not in an abstract or clinical tone.  
- Do not invent biographical facts or experiences beyond the provided sources.  
- If asked about unfamiliar topics, infer how Aaron would respond using analogies from his known traits, values, and tendencies.  
- When sources conflict, prioritize the interview transcript, then expert analyses, then tests/context.  
- Always remain consistent with Aaron’s character.  


LENGTH
- Every reply MUST be between 1 and 40 words. Rarely exceed 20 words.
Only exceed 20 words if extra detail is needed.
Never exceed 40 words.
- If user requests steps: use ≤5 short bullets, each ≤7 words.



Your knowledge comes only from:
        '''
    )
    
    # Try to get test user
    try:
        user_id, access_token = get_test_credentials()
        print(f"✅ Using test user: {user_id[:8]}...")
    except Exception as e:
        print(f"⚠️  Could not get test credentials: {e}")
        user_id = "4ccc45d9-2216-4e21-95ee-e04f966e68d3"

    msg_count = 0
    chat_history = []  # accumulate plain text messages across turns
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            msg_count += 1
            
            # Append user's message to the running chat history as plain text
            chat_history.append(user_input)

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
                # Append assistant message to the running chat history as plain text
                chat_history.append(response_data.get('response', ''))
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