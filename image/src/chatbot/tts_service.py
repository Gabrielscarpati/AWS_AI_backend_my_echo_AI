"""
Text-to-Speech service using ElevenLabs API.
"""
import os
import requests
import base64
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class ElevenLabsTTS:
    """ElevenLabs Text-to-Speech service."""

    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default voice
        self.base_url = "https://api.elevenlabs.io/v1"

        if not self.api_key:
            print("Warning: ELEVENLABS_API_KEY not found in environment variables")

    def text_to_speech(self, text: str, voice_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Convert text to speech using ElevenLabs API.

        Args:
            text: Text to convert to speech
            voice_id: Optional voice ID to use (defaults to configured voice)

        Returns:
            Dict containing 'audio_data' (bytes) and 'content_type' or None if failed
        """
        if not self.api_key:
            print("ElevenLabs API key not configured")
            return None

        if not text or not text.strip():
            print("Empty text provided for TTS")
            return None

        # Use provided voice_id or default
        target_voice_id = voice_id or self.voice_id

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }

        data = {
            "text": text.strip(),
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }

        try:
            url = f"{self.base_url}/text-to-speech/{target_voice_id}"
            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200:
                audio_data = response.content
                return {
                    "audio_data": audio_data,
                    "content_type": "audio/mpeg",
                    "size": len(audio_data)
                }
            else:
                print(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            print(f"Error calling ElevenLabs API: {e}")
            return None

    def get_available_voices(self) -> Optional[Dict[str, Any]]:
        """Get list of available voices."""
        if not self.api_key:
            return None

        headers = {
            "xi-api-key": self.api_key
        }

        try:
            url = f"{self.base_url}/voices"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting voices: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            print(f"Error getting voices: {e}")
            return None


# Global instance
tts_service = ElevenLabsTTS()


def should_generate_tts(msgs_cnt_by_user: int) -> bool:
    """Check if TTS should be generated for this message count."""
    return msgs_cnt_by_user % 15 == 0


def generate_tts_response(text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate TTS response and update state.

    Args:
        text: Text to convert to speech
        state: Current state dictionary

    Returns:
        Updated state with TTS data
    """
    msgs_cnt = state.get("msgs_cnt_by_user", 0)

    if not should_generate_tts(msgs_cnt):
        return state

    print(f"Generating TTS for message #{msgs_cnt}")

    tts_result = tts_service.text_to_speech(text)

    if tts_result:
        return {
            **state,
            "audio_output": tts_result["audio_data"],
            "audio_output_url": f"data:{tts_result['content_type']};base64,{base64.b64encode(tts_result['audio_data']).decode()}"
        }
    else:
        print("Failed to generate TTS")
        return state
