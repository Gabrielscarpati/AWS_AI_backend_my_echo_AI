"""
Multi-Provider Text-to-Speech Service

This module provides TTS integration with multiple providers:
- ElevenLabs API (recommended for production - fast, high quality)
- F5-TTS (local inference with voice cloning - fallback)
"""

import os
import random
import tempfile
import time
import requests
from typing import Dict, Any, Optional
import torch

# Environment configuration
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs")  # "elevenlabs" or "f5tts"
TTS_PROBABILITY = float(os.getenv("TTS_PROBABILITY", "0.3"))
TTS_OUTPUT_DIR = os.getenv("TTS_OUTPUT_DIR", "/tmp/tts_audio")

# ElevenLabs configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default: Rachel
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_monolingual_v1")

# F5-TTS configuration (fallback)
CUSTOM_VOICE_PATH = os.getenv("CUSTOM_VOICE_PATH")

# Lazy loading for F5-TTS to avoid import conflicts
_f5_tts_model = None
_f5_tts_available = None


def should_use_tts() -> bool:
    """Determine if TTS should be used for this response based on probability."""
    return random.random() < TTS_PROBABILITY


def prepare_output_directory():
    """Ensure the TTS output directory exists."""
    os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)


def check_elevenlabs_available() -> bool:
    """Check if ElevenLabs API is configured and available."""
    return bool(ELEVENLABS_API_KEY and ELEVENLABS_API_KEY != "your_api_key_here")


def check_f5_tts_available() -> bool:
    """Check if F5-TTS is available and can be imported."""
    global _f5_tts_available
    
    if _f5_tts_available is not None:
        return _f5_tts_available
    
    try:
        import f5_tts
        from f5_tts.api import F5TTS
        _f5_tts_available = True
        return True
    except ImportError as e:
        _f5_tts_available = False
        return False


def generate_elevenlabs_tts(text: str, custom_voice_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate TTS audio using ElevenLabs API.
    
    Args:
        text: The text to convert to speech
        custom_voice_id: Optional custom voice ID (if you've cloned a voice)
        
    Returns:
        Dictionary with generation results
    """
    start_time = time.time()
    
    if not check_elevenlabs_available():
        return {
            "success": False,
            "error": "ElevenLabs API key not configured",
            "generation_time": 0,
            "model_used": "ElevenLabs",
            "custom_voice_used": False,
            "text_length": len(text)
        }
    
    try:
        # Prepare output directory
        prepare_output_directory()
        
        # Use custom voice ID or default
        voice_id = custom_voice_id or ELEVENLABS_VOICE_ID
        
        # Generate unique filename
        timestamp = int(time.time() * 1000)
        output_filename = f"elevenlabs_output_{timestamp}.mp3"
        output_path = os.path.join(TTS_OUTPUT_DIR, output_filename)
        
        print(f"🎵 Generating ElevenLabs audio for: '{text[:50]}...'")
        print(f"🎤 Using voice ID: {voice_id}")
        
        # ElevenLabs API request
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        data = {
            "text": text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        # Make API request
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Save audio file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # Calculate generation time and duration
            generation_time = time.time() - start_time
            
            # Estimate duration (ElevenLabs typically ~150-200 words per minute)
            word_count = len(text.split())
            estimated_duration = (word_count / 175) * 60  # seconds
            
            print(f"✅ ElevenLabs audio generated in {generation_time:.2f}s: {output_path}")
            
            return {
                "success": True,
                "audio_url": output_path,
                "duration": estimated_duration,
                "generation_time": generation_time,
                "model_used": "ElevenLabs",
                "custom_voice_used": custom_voice_id is not None,
                "text_length": len(text)
            }
        else:
            error_msg = f"ElevenLabs API error: {response.status_code} - {response.text}"
            print(f"❌ {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "generation_time": time.time() - start_time,
                "model_used": "ElevenLabs",
                "custom_voice_used": False,
                "text_length": len(text)
            }
            
    except Exception as e:
        generation_time = time.time() - start_time
        error_msg = f"ElevenLabs generation failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        return {
            "success": False,
            "error": error_msg,
            "generation_time": generation_time,
            "model_used": "ElevenLabs",
            "custom_voice_used": False,
            "text_length": len(text)
        }


def _initialize_f5_tts():
    """Initialize F5-TTS model (lazy loading)."""
    global _f5_tts_model
    
    if _f5_tts_model is not None:
        return _f5_tts_model
    
    if not check_f5_tts_available():
        raise ImportError("F5-TTS is not available")
    
    try:
        from f5_tts.api import F5TTS
        print("🔄 Loading F5-TTS model...")
        
        # Initialize F5-TTS
        _f5_tts_model = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file="",
            vocab_file="",
            ode_method="euler",
            use_ema=True,
            device=None
        )
        
        print("✅ F5-TTS model loaded successfully")
        return _f5_tts_model
        
    except Exception as e:
        print(f"❌ Failed to initialize F5-TTS: {e}")
        _f5_tts_model = None
        raise


def generate_f5_tts(text: str, custom_voice_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate TTS audio using F5-TTS (local inference).
    
    Args:
        text: The text to convert to speech
        custom_voice_path: Optional path to custom voice file for cloning
        
    Returns:
        Dictionary with generation results
    """
    start_time = time.time()
    
    try:
        # Initialize F5-TTS model
        model = _initialize_f5_tts()
        
        # Prepare output directory
        prepare_output_directory()
        
        # Use custom voice or default
        voice_path = custom_voice_path or CUSTOM_VOICE_PATH
        
        # Generate unique filename
        timestamp = int(time.time() * 1000)
        output_filename = f"f5_tts_output_{timestamp}.wav"
        output_path = os.path.join(TTS_OUTPUT_DIR, output_filename)
        
        print(f"🎵 Generating F5-TTS audio for: '{text[:50]}...'")
        
        # Generate audio with F5-TTS
        if voice_path and os.path.exists(voice_path):
            print(f"🎤 Using custom voice: {voice_path}")
            result = model.infer(
                ref_file=voice_path,
                ref_text="",
                gen_text=text,
                remove_silence=True,
                cfg_strength=3.0,
                nfe_step=64,
                speed=0.9
            )
        else:
            print("🎤 Using default voice")
            import sys
            venv_path = sys.executable.replace('/bin/python', '')
            default_ref_file = os.path.join(venv_path, 'lib', 'python3.10', 'site-packages', 'f5_tts', 'infer', 'examples', 'basic', 'basic_ref_en.wav')
            
            if os.path.exists(default_ref_file):
                result = model.infer(
                    ref_file=default_ref_file,
                    ref_text="Some call me nature, others call me mother nature.",
                    gen_text=text,
                    remove_silence=True,
                    cfg_strength=3.0,
                    nfe_step=64,
                    speed=0.95
                )
            else:
                raise FileNotFoundError(f"Default reference file not found: {default_ref_file}")
        
        # Handle the result
        if isinstance(result, tuple) and len(result) >= 2:
            audio, sample_rate = result[0], result[1]
        else:
            raise ValueError(f"Unexpected F5-TTS result format")
        
        # Save the audio to file
        import soundfile as sf
        sf.write(output_path, audio, sample_rate)
        
        # Calculate generation time and duration
        generation_time = time.time() - start_time
        word_count = len(text.split())
        estimated_duration = (word_count / 150) * 60
        
        print(f"✅ F5-TTS audio generated in {generation_time:.2f}s: {output_path}")
        
        return {
            "success": True,
            "audio_url": output_path,
            "duration": estimated_duration,
            "generation_time": generation_time,
            "model_used": "F5-TTS",
            "custom_voice_used": voice_path is not None,
            "text_length": len(text)
        }
        
    except Exception as e:
        generation_time = time.time() - start_time
        error_msg = f"F5-TTS generation failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        return {
            "success": False,
            "error": error_msg,
            "generation_time": generation_time,
            "model_used": "F5-TTS",
            "custom_voice_used": False,
            "text_length": len(text)
        }


def generate_tts_response(text: str, custom_voice_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate TTS audio using the configured provider with fallback.
    
    Args:
        text: The text to convert to speech
        custom_voice_url: Optional custom voice (ElevenLabs voice ID or F5-TTS file path)
        
    Returns:
        Dictionary with generation results
    """
    print(f"🎵 TTS Provider: {TTS_PROVIDER}")
    
    # Try primary provider first
    if TTS_PROVIDER == "elevenlabs" and check_elevenlabs_available():
        result = generate_elevenlabs_tts(text, custom_voice_url)
        if result.get("success"):
            return result
        else:
            print("⚠️ ElevenLabs failed, falling back to F5-TTS...")
    
    # Fallback to F5-TTS or if F5-TTS is primary
    if check_f5_tts_available():
        return generate_f5_tts(text, custom_voice_url)
    else:
        return {
            "success": False,
            "error": "No TTS provider available",
            "generation_time": 0,
            "model_used": "None",
            "custom_voice_used": False,
            "text_length": len(text)
        }


def get_system_info() -> Dict[str, Any]:
    """Get system information for TTS setup."""
    return {
        "tts_provider": TTS_PROVIDER,
        "elevenlabs_available": check_elevenlabs_available(),
        "f5_tts_available": check_f5_tts_available(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "tts_probability": TTS_PROBABILITY,
        "output_directory": TTS_OUTPUT_DIR,
        "custom_voice_path": CUSTOM_VOICE_PATH,
        "elevenlabs_voice_id": ELEVENLABS_VOICE_ID
    }


# Main function for testing
if __name__ == "__main__":
    print("🧪 Testing Multi-Provider TTS Service")
    
    # Print system info
    info = get_system_info()
    print("\n📊 System Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test TTS generation
    test_text = "The sun dipped below the horizon, painting the sky in warm shades of orange and pink. A gentle breeze carried the scent of blooming flowers across the quiet field. In that moment, everything felt calm and timeless."
    print(f"\n🎵 Testing TTS generation with text: '{test_text}'")
    
    result = generate_tts_response(test_text)
    print(f"\n📋 Generation Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
        
    if result.get("success"):
        audio_file = result.get("audio_url")
        if audio_file and os.path.exists(audio_file):
            print(f"\n🎧 Opening audio file: {audio_file}")
            import subprocess
            try:
                subprocess.run(["open", audio_file], check=True)
            except:
                print(f"Could not open audio file automatically. File location: {audio_file}")