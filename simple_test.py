#!/usr/bin/env python3
"""
Simple test script to quickly test the chatbot functionality.
"""

import sys
import os
import json
from pathlib import Path

# Add the image/src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "image" / "src"))

# Change to the image/src directory
os.chdir(Path(__file__).parent / "image" / "src")

from app import handler

def simple_test():
    """Run a simple test of the chatbot."""
    print("🧪 Running simple chatbot test...")
    
    # Create a simple test event
    test_message = "Hello, I'm feeling a bit stressed today. Can you help me?"
    
    event = {
        "body": json.dumps({
            "user_id": "test-user-123",
            "creator_id": None,  # Optional
            "chat_history": [
                ("user", test_message)
            ],
            "msgs_cnt_by_user": 1
        }),
        "isBase64Encoded": False
    }
    
    print(f"📝 Test message: {test_message}")
    print("🤔 Processing...")
    
    try:
        response = handler(event, {})
        
        if response["statusCode"] == 200:
            response_data = json.loads(response["body"])
            print(f"✅ Success!")
            print(f"🤖 Haven's response: {response_data['response']}")
        else:
            error_data = json.loads(response["body"])
            print(f"❌ Error {response['statusCode']}: {error_data.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()

