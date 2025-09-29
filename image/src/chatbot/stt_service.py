"""
Speech-to-Text service using Assembly AI API.
"""
import os
import base64
import tempfile
from typing import Optional
from dotenv import load_dotenv
import assemblyai as aai

load_dotenv()


class AssemblyAISTT:
    """Assembly AI Speech-to-Text service."""

    def __init__(self):
        self.api_key = os.getenv("ASSEMBLY_AI_API_KEY")
        
        if not self.api_key:
            print("Warning: ASSEMBLY_AI_API_KEY not found in environment variables")
        else:
            aai.settings.api_key = self.api_key
            self.transcriber = aai.Transcriber()

    def speech_to_text(self, audio_data: bytes, language: str = "en") -> Optional[str]:
        """
        Convert audio to text using Assembly AI API.

        Args:
            audio_data: Raw audio file bytes (not base64 encoded)
            language: Language code (default: 'en')

        Returns:
            Transcribed text or None if failed
        """
        if not self.api_key:
            print("Assembly AI API key not configured")
            return None

        if not audio_data:
            print("Empty audio data provided for STT")
            return None

        try:
            # Create a temporary file to store the audio data
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            # Transcribe the audio file
            transcript = self.transcriber.transcribe(temp_file_path)
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
            # Check if transcription was successful
            if transcript.status == aai.TranscriptStatus.completed:
                transcribed_text = transcript.text.strip()
                if transcribed_text:
                    print(f"✅ Audio transcribed successfully: '{transcribed_text[:100]}...'")
                    return transcribed_text
                else:
                    print("Empty transcription result")
                    return None
            elif transcript.status == aai.TranscriptStatus.error:
                print(f"Assembly AI transcription error: {transcript.error}")
                return None
            else:
                print(f"Assembly AI transcription failed with status: {transcript.status}")
                return None

        except Exception as e:
            print(f"Error calling Assembly AI API: {e}")
            # Clean up temp file if it exists
            try:
                if 'temp_file_path' in locals():
                    os.unlink(temp_file_path)
            except:
                pass
            return None


# Global instance
stt_service = AssemblyAISTT()


def transcribe_audio(audio_data: bytes) -> Optional[str]:
    """
    Transcribe audio data to text.

    Args:
        audio_data: Raw audio file bytes (not base64 encoded)

    Returns:
        Transcribed text or None if failed
    """
    return stt_service.speech_to_text(audio_data)