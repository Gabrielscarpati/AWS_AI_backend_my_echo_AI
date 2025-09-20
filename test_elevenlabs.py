#!/usr/bin/env python3
"""
Test ElevenLabs TTS Integration

This script tests the ElevenLabs TTS functionality with fallback to F5-TTS.
"""

import os
import sys

# Add the image/src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'image', 'src'))

def test_elevenlabs_tts():
    """Test the ElevenLabs TTS integration."""
    print("🎤 Testing ElevenLabs TTS Integration")
    print("=" * 60)
    
    # Set test environment for ElevenLabs
    os.environ['TTS_PROVIDER'] = 'elevenlabs'
    os.environ['TTS_PROBABILITY'] = '1.0'  # Force TTS generation
    os.environ['TTS_OUTPUT_DIR'] = '/tmp/tts_audio'
    
    # Note: You'll need to set your actual API key
    if not os.getenv('ELEVENLABS_API_KEY'):
        os.environ['ELEVENLABS_API_KEY'] = 'your_api_key_here'  # Replace with real key
        print("⚠️  Please set your ELEVENLABS_API_KEY environment variable")
        print("   You can get one from: https://elevenlabs.io/")
    
    try:
        from chatbot.tts_service import generate_tts_response, get_system_info
        
        # Get system information
        info = get_system_info()
        print("📊 System Information:")
        for key, value in info.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
        
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
                print(f"   Generation time: {result.get('generation_time', 0):.2f}s")
                print(f"   Model used: {result.get('model_used')}")
                
                # Try to open the audio file
                try:
                    import subprocess
                    subprocess.run(["open", audio_path], check=True)
                    print(f"🎧 Opening audio file...")
                except:
                    print(f"🎧 Audio file ready at: {audio_path}")
                
                return True
            else:
                print(f"❌ Audio file not found: {audio_path}")
                return False
        else:
            print(f"❌ TTS generation failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing ElevenLabs TTS: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_behavior():
    """Test the fallback from ElevenLabs to F5-TTS."""
    print("\n🔄 Testing Fallback Behavior")
    print("=" * 60)
    
    # Temporarily disable ElevenLabs to test fallback
    original_key = os.environ.get('ELEVENLABS_API_KEY')
    os.environ['ELEVENLABS_API_KEY'] = 'invalid_key'
    
    try:
        from chatbot.tts_service import generate_tts_response
        
        test_text = "This should fallback to F5-TTS since ElevenLabs key is invalid."
        print(f"🎵 Testing fallback with invalid ElevenLabs key...")
        print(f"Text: '{test_text}'")
        
        result = generate_tts_response(test_text)
        
        print(f"\n📋 Fallback Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        success = result.get("success")
        model_used = result.get("model_used")
        
        if success and model_used == "F5-TTS":
            print("✅ Fallback to F5-TTS successful!")
        elif success and model_used == "ElevenLabs":
            print("⚠️  Still using ElevenLabs (unexpected)")
        else:
            print("❌ Fallback failed")
            
    except Exception as e:
        print(f"❌ Error testing fallback: {e}")
    finally:
        # Restore original API key
        if original_key:
            os.environ['ELEVENLABS_API_KEY'] = original_key
        else:
            os.environ.pop('ELEVENLABS_API_KEY', None)


def main():
    """Run all tests."""
    print("🚀 ElevenLabs TTS Integration Test Suite")
    print("=" * 70)
    
    # Test 1: ElevenLabs TTS
    elevenlabs_success = test_elevenlabs_tts()
    
    # Test 2: Fallback behavior
    test_fallback_behavior()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary:")
    print(f"  ElevenLabs TTS: {'✅ PASS' if elevenlabs_success else '❌ FAIL'}")
    
    if elevenlabs_success:
        print("\n🎉 ElevenLabs integration is working!")
        print("\n📝 Next steps:")
        print("1. Get your ElevenLabs API key from https://elevenlabs.io/")
        print("2. Clone Deija's voice using your sample audio")
        print("3. Update your .env file with the API key and voice ID")
        print("4. Set TTS_PROVIDER=elevenlabs in your .env")
    else:
        print("\n⚠️  ElevenLabs integration needs configuration.")
        print("Make sure to set your ELEVENLABS_API_KEY environment variable.")


if __name__ == "__main__":
    main()
