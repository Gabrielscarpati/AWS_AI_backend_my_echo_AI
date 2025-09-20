"""
F5-TTS Text-to-Speech Service

This module provides F5-TTS integration for generating audio responses with voice cloning capabilities.
"""

import os
import random
import tempfile
import time
from typing import Dict, Any, Optional
import torch

# Environment configuration
TTS_PROBABILITY = float(os.getenv("TTS_PROBABILITY", "0.3"))
TTS_OUTPUT_DIR = os.getenv("TTS_OUTPUT_DIR", "/tmp/tts_audio")
CUSTOM_VOICE_PATH = os.getenv("CUSTOM_VOICE_PATH")

# Lazy loading for F5-TTS to avoid import conflicts
_f5_tts_model = None
_f5_tts_available = None


def check_f5_tts_available() -> bool:
    """Check if F5-TTS is available and can be imported."""
    global _f5_tts_available
    
    if _f5_tts_available is not None:
        return _f5_tts_available
    
    try:
        import f5_tts
        from f5_tts.api import F5TTS
        _f5_tts_available = True
        print("✅ F5-TTS is available")
        return True
    except ImportError as e:
        _f5_tts_available = False
        print(f"❌ F5-TTS not available: {e}")
        return False


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
            model="F5TTS_v1_Base",  # Default model
            ckpt_file="",  # Uses default checkpoint
            vocab_file="",  # Uses default vocab
            ode_method="euler",
            use_ema=True,
            device=None  # Auto-detect device
        )
        
        print("✅ F5-TTS model loaded successfully")
        return _f5_tts_model
        
    except Exception as e:
        print(f"❌ Failed to initialize F5-TTS: {e}")
        _f5_tts_model = None
        raise


def should_use_tts() -> bool:
    """Determine if TTS should be used for this response based on probability."""
    return random.random() < TTS_PROBABILITY


def prepare_output_directory():
    """Ensure the TTS output directory exists."""
    os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)


def generate_tts_response(text: str, custom_voice_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate TTS audio using F5-TTS.
    
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
            # Generate with voice cloning
            result = model.infer(
                ref_file=voice_path,
                ref_text="",  # F5-TTS can work without reference text
                gen_text=text,
                remove_silence=True,
                cfg_strength=3.0,    # Higher = more similar to reference voice (default: 2.0)
                nfe_step=64,         # Higher = better quality but slower (default: 32)
                speed=0.9           # Slightly slower for better clarity (default: 1.0)
            )
        else:
            print("🎤 Using default voice (no custom voice provided)")
            # Use the built-in English reference file (hardcoded path for now)
            import sys
            venv_path = sys.executable.replace('/bin/python', '')
            default_ref_file = os.path.join(venv_path, 'lib', 'python3.10', 'site-packages', 'f5_tts', 'infer', 'examples', 'basic', 'basic_ref_en.wav')
            
            if os.path.exists(default_ref_file):
                print(f"🎤 Using default reference: {default_ref_file}")
                result = model.infer(
                    ref_file=default_ref_file,
                    ref_text="Some call me nature, others call me mother nature.",
                    gen_text=text,
                    remove_silence=True,
                    cfg_strength=3.0,    # Higher = more similar to reference voice (default: 2.0)
                    nfe_step=64,         # Higher = better quality but slower (default: 32)
                    speed=0.95           # Slightly slower for better clarity (default: 1.0)
                )
            else:
                print(f"❌ Default reference file not found: {default_ref_file}")
                raise FileNotFoundError(f"Default reference file not found: {default_ref_file}")
        
        # Handle the result - it might return more than 2 values
        if isinstance(result, tuple):
            if len(result) == 2:
                audio, sample_rate = result
            elif len(result) >= 2:
                audio, sample_rate = result[0], result[1]
                print(f"ℹ️  F5-TTS returned {len(result)} values, using first 2")
            else:
                raise ValueError(f"Unexpected F5-TTS result format: {len(result)} values")
        else:
            raise ValueError(f"Unexpected F5-TTS result type: {type(result)}")
        
        # Save the audio to file
        import soundfile as sf
        sf.write(output_path, audio, sample_rate)
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Get audio duration (approximate based on text length)
        # F5-TTS typically generates ~150 words per minute
        word_count = len(text.split())
        estimated_duration = (word_count / 150) * 60  # seconds
        
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


def get_system_info() -> Dict[str, Any]:
    """Get system information for TTS setup."""
    return {
        "f5_tts_available": check_f5_tts_available(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "tts_probability": TTS_PROBABILITY,
        "output_directory": TTS_OUTPUT_DIR,
        "custom_voice_path": CUSTOM_VOICE_PATH,
        "custom_voice_exists": CUSTOM_VOICE_PATH and os.path.exists(CUSTOM_VOICE_PATH)
    }


# Main function for testing
if __name__ == "__main__":
    print("🧪 Testing F5-TTS Service")
    
    # Print system info
    info = get_system_info()
    print("\n📊 System Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test TTS generation
    if info["f5_tts_available"]:
        test_text = "The sun dipped below the horizon, painting the sky in warm shades of orange and pink. A gentle breeze carried the scent of blooming flowers across the quiet field. In that moment, everything felt calm and timeless."
        print(f"\n🎵 Testing TTS generation with text: '{test_text}'")
        
        result = generate_tts_response(test_text)
        print(f"\n📋 Generation Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("\n❌ Cannot test TTS generation - F5-TTS not available")
