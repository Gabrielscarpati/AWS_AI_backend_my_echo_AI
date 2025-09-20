#!/usr/bin/env python3
"""
Test script for F5-TTS integration

This script tests the F5-TTS service and the complete chatbot integration.
"""

import os
import sys
import tempfile

# Add the image/src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'image', 'src'))

def test_f5_tts_service():
    """Test the F5-TTS service directly."""
    print("🧪 Testing F5-TTS Service")
    print("=" * 50)
    
    try:
        from chatbot.tts_service import get_system_info, generate_tts_response
        
        # Get system information
        info = get_system_info()
        print("📊 System Information:")
        for key, value in info.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
        
        if not info["f5_tts_available"]:
            print("\n❌ F5-TTS not available. Please install with: ./venv/bin/pip install f5-tts")
            return False
        
        # Test TTS generation
        test_text = "The sun dipped below the horizon, painting the sky in warm shades of orange and pink. A gentle breeze carried the scent of blooming flowers across the quiet field. In that moment, everything felt calm and timeless."
        print(f"\n🎵 Testing TTS generation...")
        print(f"Text: '{test_text}'")
        
        result = generate_tts_response(test_text)
        
        print(f"\n📋 Generation Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        if result.get("success"):
            audio_path = result.get("audio_url")
            if audio_path and os.path.exists(audio_path):
                print(f"✅ Audio file generated successfully: {audio_path}")
                file_size = os.path.getsize(audio_path)
                print(f"   File size: {file_size} bytes")
                return True
            else:
                print(f"❌ Audio file not found: {audio_path}")
                return False
        else:
            print(f"❌ TTS generation failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing F5-TTS service: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chatbot_integration():
    """Test the complete chatbot integration with TTS."""
    print("\n🤖 Testing Chatbot Integration")
    print("=" * 50)
    
    try:
        from app import handler
        
        # Create a test event
        test_event = {
            "body": """{
                "user_id": "test_user",
                "creator_id": "test_creator",
                "influencer_name": "Test Assistant",
                "chat_history": [["user", "Hello, can you tell me a short joke?"]],
                "msgs_cnt_by_user": 1
            }""",
            "isBase64Encoded": False
        }
        
        print("📤 Sending test request to chatbot...")
        result = handler(test_event, {})
        
        status_code = result.get("statusCode", 500)
        print(f"📥 Response status: {status_code}")
        
        if status_code == 200:
            import json
            body = json.loads(result.get("body", "{}"))
            
            print("📋 Response fields:")
            for key, value in body.items():
                if key in ["response", "response_type", "audio_url", "audio_duration"]:
                    print(f"  {key}: {value}")
            
            response_type = body.get("response_type", "text")
            if response_type == "audio":
                print("🎵 Audio response generated!")
                audio_url = body.get("audio_url")
                if audio_url and os.path.exists(audio_url):
                    print(f"✅ Audio file exists: {audio_url}")
                    return True
                else:
                    print(f"❌ Audio file missing: {audio_url}")
                    return False
            else:
                print("📝 Text response generated (TTS not triggered this time)")
                return True
        else:
            print(f"❌ Request failed with status {status_code}")
            print(f"Response: {result.get('body')}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing chatbot integration: {e}")
        import traceback
        traceback.print_exc()
        return False


def setup_test_environment():
    """Set up environment variables for testing."""
    print("⚙️ Setting up test environment...")
    
    # Set basic TTS configuration
    os.environ["TTS_PROBABILITY"] = "1.0"  # Force TTS generation for testing
    os.environ["TTS_OUTPUT_DIR"] = tempfile.gettempdir()
    
    # Set required API keys to dummy values if not present
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "dummy_key_for_testing"
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = "dummy_key_for_testing"
    
    print(f"✅ TTS output directory: {os.environ['TTS_OUTPUT_DIR']}")
    print(f"✅ TTS probability: {os.environ['TTS_PROBABILITY']}")


def main():
    """Run all tests."""
    print("🚀 F5-TTS Integration Test Suite")
    print("=" * 60)
    
    setup_test_environment()
    
    # Test F5-TTS service
    service_success = test_f5_tts_service()
    
    # Only test integration if service works
    if service_success:
        integration_success = test_chatbot_integration()
    else:
        integration_success = False
        print("\n⏭️  Skipping integration test due to service failure")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"  F5-TTS Service: {'✅ PASS' if service_success else '❌ FAIL'}")
    print(f"  Integration:    {'✅ PASS' if integration_success else '❌ FAIL'}")
    
    if service_success and integration_success:
        print("\n🎉 All tests passed! F5-TTS integration is working.")
    elif service_success:
        print("\n⚠️  F5-TTS service works, but integration needs attention.")
    else:
        print("\n❌ F5-TTS service failed. Check installation and configuration.")
    
    return service_success and integration_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
