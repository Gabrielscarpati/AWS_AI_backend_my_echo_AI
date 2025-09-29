"""
Text-to-Speech service using Fish AI API.
"""
import os
import base64
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from fish_audio_sdk import Session, TTSRequest

load_dotenv()


class FishAITTS:
    """Fish AI Text-to-Speech service."""

    def __init__(self):
        self.api_key = os.getenv("FISH_AI_API_KEY")
        
        if not self.api_key:
            print("Warning: FISH_AI_API_KEY not found in environment variables")
        else:
            self.session = Session(self.api_key)

    def text_to_speech(self, text: str, voice_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Convert text to speech using Fish AI API.

        Args:
            text: Text to convert to speech
            voice_id: Optional voice model ID to use

        Returns:
            Dict containing 'audio_data' (bytes) and 'content_type' or None if failed
        """
        if not self.api_key:
            print("Fish AI API key not configured")
            return None

        if not text or not text.strip():
            print("Empty text provided for TTS")
            return None

        try:
            # Create a TTS request
            request = TTSRequest(
                text=text.strip(),
                reference_id=voice_id,  # Optional: specify a voice model ID
                format="mp3"  # Output format
            )

            # Generate audio data
            audio_chunks = []
            for chunk in self.session.tts(request):
                audio_chunks.append(chunk)
            
            audio_data = b''.join(audio_chunks)
            
            if audio_data:
                return {
                    "audio_data": audio_data,
                    "content_type": "audio/mpeg",
                    "size": len(audio_data)
                }
            else:
                print("No audio data received from Fish AI")
                return None

        except Exception as e:
            print(f"Error calling Fish AI API: {e}")
            return None

    def get_available_voices(self) -> Optional[Dict[str, Any]]:
        """Get list of available voices (placeholder for Fish AI)."""
        if not self.api_key:
            return None
        
        # Fish AI doesn't provide a direct voice listing endpoint in the basic SDK
        # This would need to be implemented based on their specific API documentation
        print("Voice listing not implemented for Fish AI - refer to Fish AI documentation for available voice models")
        return None


# Global instance
tts_service = FishAITTS()


def generate_tts_response(text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate TTS response and update state.

    Args:
        text: Text to convert to speech
        state: Current state dictionary

    Returns:
        Updated state with TTS data
    """
    # Check if TTS should be generated based on API field only
    should_generate = state.get("should_generate_tts", False)

    if not should_generate:
        return state

    print("Generating TTS based on API request")

    # Get voice ID from state, fallback to None if not provided
    voice_id = state.get("fish_ai_voice_id")
    tts_result = tts_service.text_to_speech(text, voice_id=voice_id)

    if tts_result:
        return {
            **state,
            "audio_output": tts_result["audio_data"],
            "audio_output_url": f"data:{tts_result['content_type']};base64,{base64.b64encode(tts_result['audio_data']).decode()}"
        }
    else:
        print("Failed to generate TTS")
        return state