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

def create_test_event(user_message: str, user_id: str = None, creator_id: str = None):
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
    
    # Simple chat history with user message
    chat_history = [
        ("user", user_message)
    ]
    
    event = {
        "body": json.dumps({
            "user_id": user_id,
            "creator_id": creator_id,
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
    creator_id = None
    
    # Try to get test user
    try:
        user_id, access_token = get_test_credentials()
        print(f"✅ Using test user: {user_id[:8]}...")
    except Exception as e:
        print(f"⚠️  Could not get test credentials: {e}")
        user_id = "test-user-local"
    
    # Ask for creator_id (optional)
    creator_input = input("Enter creator_id (optional, press Enter to skip): ").strip()
    if creator_input:
        creator_id = creator_input
    
    msg_count = 0
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            msg_count += 1
            
            # Create test event
            event = create_test_event(user_input, user_id, creator_id)
            event["body"] = json.loads(event["body"])
            event["body"]["msgs_cnt_by_user"] = msg_count
            event["body"] = json.dumps(event["body"])
            
            print("🤔 Haven is thinking...")
            
            # Call the handler
            response = handler(event, {})
            
            if response["statusCode"] == 200:
                response_data = json.loads(response["body"])
                print(f"🤖 Haven: {response_data['response']}")
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
