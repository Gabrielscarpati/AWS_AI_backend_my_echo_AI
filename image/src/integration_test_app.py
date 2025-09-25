#!/usr/bin/env python3
"""
Local runner for the AWS Lambda handler.
This allows running the app.py handler locally without needing AWS Lambda.
"""
import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to Python path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import handler


def create_event(user_message="What's your name?", **kwargs):
    prompt ='''
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
    defaults = {
        "user_id": "test_user_123",
        "creator_id": "aaron_ai",
        "influencer_name": "Aaron",
        "influencer_personality_prompt": prompt,
        "chat_history": [user_message],
        "msgs_cnt_by_user": 1,
        "input_media_type": "image",
        "image_data":"iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==",
        "user_query": user_message,
        "should_generate_tts": True
    }

    # Override defaults with any provided kwargs
    defaults.update(kwargs)

    return {
        "body": json.dumps(defaults),
        "isBase64Encoded": False
    }

def run_handler(user_message="What's your name?", **kwargs):
    """Run the Lambda handler with the given message and options."""
    print("🚀 Starting local handler execution...")
    print("=" * 50)

    # Check required environment variables
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False

    print("✅ Environment variables loaded")

    # Create event
    event = create_event(user_message, **kwargs)

    print("📥 Request:")
    print(json.dumps(json.loads(event["body"]), indent=2))
    print("\n" + "="*50)

    try:
        # Run the handler
        result = handler(event, {})

        print("📤 Response:")
        if result["statusCode"] == 200:
            response_data = json.loads(result["body"])
            print(json.dumps(response_data, indent=2))

            # Show additional info
            if response_data.get("audio_output"):
                print(f"\n🔊 Audio generated: {len(response_data['audio_output'])} bytes")

            if response_data.get("timings"):
                timings = response_data["timings"]
                total = sum(v for v in timings.values() if isinstance(v, (int, float)))
                print(f"\n⏱️  Total execution time: {total}s")
        else:
            error_data = json.loads(result["body"])
            print(f"❌ Error: {error_data.get('error', 'Unknown error')}")

        return result["statusCode"] == 200

    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point when run as a script."""
    # Check if a message was provided as command line argument
    user_message = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hi, how are you?"

    # Run the handler
    success = run_handler(user_message)

    if success:
        print("\n✅ Handler executed successfully!")
    else:
        print("\n❌ Handler execution failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()