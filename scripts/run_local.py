#!/usr/bin/env python3
"""Run a local test event against the Lambda handler and print structured logs.

Usage:
  python scripts/run_local.py
Optional env vars are read from image/src/.env via python-dotenv if present.
"""
import sys, os
# Add both repo root and image/src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "image", "src"))

import json
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "image", "src", ".env"))

def main():
    # Change working directory to image/src so relative file paths work
    original_cwd = os.getcwd()
    image_src_dir = os.path.join(os.path.dirname(__file__), "..", "image", "src")
    os.chdir(image_src_dir)
    
    try:
        from app import handler

        event = {
        "body": json.dumps({
            "user_id": os.getenv("TEST_USER_ID", "test_user"),
            "creator_id": os.getenv("TEST_CREATOR_ID", "taylor_swift"),
            "chat_history": [["user", "Hi, how are you?"]],
            "msgs_cnt_by_user": 1
        }),
        "isBase64Encoded": False
    }

    print("Running local handler with event:")
    print(json.dumps(json.loads(event["body"]), indent=2))
    print("\n" + "="*50)
    
        try:
            res = handler(event, {})
            print("Handler response:")
            print(json.dumps(json.loads(res["body"]), indent=2))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()
