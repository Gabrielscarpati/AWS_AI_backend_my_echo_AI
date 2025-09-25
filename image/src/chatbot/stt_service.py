"""
Speech-to-Text service using OpenAI Whisper API.
"""
import os
import base64
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class OpenAIWhisperSTT:
    """OpenAI Whisper Speech-to-Text service."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"

        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")

    def speech_to_text(self, audio_data: bytes, language: str = "en") -> Optional[str]:
        """
        Convert audio to text using OpenAI Whisper API.

        Args:
            audio_data: Raw audio file bytes (not base64 encoded)
            language: Language code (default: 'en')

        Returns:
            Transcribed text or None if failed
        """
        if not self.api_key:
            print("OpenAI API key not configured")
            return None

        if not audio_data:
            print("Empty audio data provided for STT")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        # Create a file-like object from the bytes
        from io import BytesIO
        audio_file = BytesIO(audio_data)
        audio_file.name = "audio.mp3"  # Set a name for the file

        # Prepare the audio file data - let Whisper auto-detect format
        files = {
            "file": ("audio.mp3", audio_file, "audio/mp3")
        }

        data = {
            "model": "whisper-1",
            "language": language,
            "response_format": "json"
        }

        try:
            url = f"{self.base_url}/audio/transcriptions"
            response = requests.post(url, headers=headers, files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                transcribed_text = result.get("text", "").strip()
                if transcribed_text:
                    print(f"✅ Audio transcribed successfully: '{transcribed_text[:100]}...'")
                    return transcribed_text
                else:
                    print("Empty transcription result")
                    return None
            else:
                print(f"Whisper API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            print(f"Error calling Whisper API: {e}")
            return None


# Global instance
stt_service = OpenAIWhisperSTT()


def transcribe_audio(audio_data: bytes) -> Optional[str]:
    """
    Transcribe audio data to text.

    Args:
        audio_data: Raw audio file bytes (not base64 encoded)

    Returns:
        Transcribed text or None if failed
    """
    return stt_service.speech_to_text(audio_data)
