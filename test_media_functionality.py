#!/usr/bin/env python3
"""
Test script for media functionality (audio/image input and TTS output).
This demonstrates how to use the new media processing features.
"""

import sys
import os
import json
import base64
from pathlib import Path

# Add the image/src directory to Python path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent / "image" / "src"))

# Change to the image/src directory so relative imports work
os.chdir(Path(__file__).parent / "image" / "src")

from app import handler


def test_text_input():
    """Test basic text input functionality."""
    print("🧪 Testing text input...")

    event = {
        "body": json.dumps({
            "user_id": "test_user_123",
            "creator_id": "aaron_ai",
            "influencer_name": "mihir",
            "influencer_personality_prompt": "You are Mihir. Speak with intelligence and expertise.",
            "chat_history": ["Hello, how are you?"],
            "msgs_cnt_by_user": 1,
            "input_media_type": "text",
            "user_query": "Hello, how are you?",
            "should_generate_tts": False
        }),
        "isBase64Encoded": False
    }

    response = handler(event, {})
    print(f"Status: {response['statusCode']}")
    if response['statusCode'] == 200:
        data = json.loads(response['body'])
        print(f"Response: {data['response'][:100]}...")
        print(f"Has audio output: {'audio_output' in data}")
    print()


def test_audio_input_simulation():
    """Test audio input simulation (using text as placeholder)."""
    print("🧪 Testing audio input simulation...")

    # For testing, we'll simulate audio data with a simple base64 string
    # In real usage, this would be actual audio data
    fake_audio_data = base64.b64encode(b"fake audio data for testing").decode()

    event = {
        "body": json.dumps({
            "user_id": "test_user_123",
            "creator_id": "aaron_ai",
            "influencer_name": "mihir",
            "influencer_personality_prompt": "You are Mihir. Speak with intelligence and expertise.",
            "chat_history": [],
            "msgs_cnt_by_user": 15,
            "input_media_type": "audio",
            "audio_data": fake_audio_data,
            "user_query": "Hello from audio",  # Fallback text
            "should_generate_tts": True  # Enable TTS for this test
        }),
        "isBase64Encoded": False
    }

    response = handler(event, {})
    print(f"Status: {response['statusCode']}")
    if response['statusCode'] == 200:
        data = json.loads(response['body'])
        print(f"Response: {data['response'][:100]}...")
        print(f"Has audio output: {'audio_output' in data}")
        print(f"Input media type: {data.get('input_media_type')}")
    print()


def test_image_input_simulation():
    """Test image input simulation (using text as placeholder)."""
    print("🧪 Testing image input simulation...")

    # For testing, we'll simulate image data with a simple base64 string
    # In real usage, this would be actual image data
    fake_image_data = base64.b64encode(b"fake image data for testing").decode()

    event = {
        "body": json.dumps({
            "user_id": "test_user_123",
            "creator_id": "aaron_ai",
            "influencer_name": "mihir",
            "influencer_personality_prompt": "You are Mihir. Speak with intelligence and expertise.",
            "chat_history": [],
            "msgs_cnt_by_user": 1,
            "input_media_type": "image",
            "image_data": fake_image_data,
            "user_query": "Describe this image",  # Query for the image
            "should_generate_tts": False  # Disable TTS for image test
        }),
        "isBase64Encoded": False
    }

    response = handler(event, {})
    print(f"Status: {response['statusCode']}")
    if response['statusCode'] == 200:
        data = json.loads(response['body'])
        print(f"Response: {data['response'][:100]}...")
        print(f"Has audio output: {'audio_output' in data}")
        print(f"Input media type: {data.get('input_media_type')}")
        print(f"Image description: {data.get('image_description', 'N/A')[:100]}...")
    print()


def main():
    """Run all media functionality tests."""
    print("🎵 Testing Media Functionality")
    print("=" * 50)

    # Check environment variables
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file to run the tests.")
        return

    print("✅ Environment variables loaded")

    # Run tests
    try:
        test_text_input()
        test_audio_input_simulation()
        test_image_input_simulation()

        print("✅ All tests completed!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
