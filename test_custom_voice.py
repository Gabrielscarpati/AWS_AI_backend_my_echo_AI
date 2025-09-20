#!/usr/bin/env python3
"""Quick test for custom voice cloning with F5-TTS"""

import os
import sys

# Set environment variables
os.environ['CUSTOM_VOICE_PATH'] = './sample_audio_deija.wav'
os.environ['TTS_OUTPUT_DIR'] = '/tmp/tts_audio'
os.environ['TTS_PROBABILITY'] = '1.0'

# Add path
sys.path.insert(0, 'image/src')

print("🎤 Testing F5-TTS with Deija's custom voice...")
print(f"Voice file: {os.environ['CUSTOM_VOICE_PATH']}")

try:
    from chatbot.tts_service import generate_tts_response
    
    text = "The sun dipped below the horizon, painting the sky in warm shades of orange and pink. A gentle breeze carried the scent of blooming flowers across the quiet field. In that moment, everything felt calm and timeless."
    print(f"Text: {text}")
    
    result = generate_tts_response(text)
    
    if result.get('success'):
        print(f"✅ Success! Audio saved to: {result.get('audio_url')}")
        print(f"Custom voice used: {result.get('custom_voice_used')}")
        print(f"Generation time: {result.get('generation_time'):.1f}s")
        
        # Try to open the file
        audio_file = result.get('audio_url')
        if audio_file and os.path.exists(audio_file):
            print(f"🎵 Opening audio file...")
            os.system(f"open '{audio_file}'")
        else:
            print(f"❌ Audio file not found: {audio_file}")
    else:
        print(f"❌ Failed: {result.get('error')}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
