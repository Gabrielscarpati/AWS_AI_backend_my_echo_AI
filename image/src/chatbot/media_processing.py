"""
Media processing functions for audio and image inputs.
"""
import base64
from typing import Dict, Any
from .state import State
from .stt_service import transcribe_audio
from .image_service import process_image_input
from .tts_service import generate_tts_response
from PIL import Image
import io


def process_media_input(state: State) -> State:
    """
    Process media input (audio or image) and convert to text for database querying.

    Args:
        state: Current state with potential media data

    Returns:
        Updated state with processed text query
    """
    input_media_type = state.get("input_media_type", "text")
    user_query = state.get("user_query", "")

    print(f"Processing media input type: {input_media_type}")

    if input_media_type == "audio" and state.get("audio_data"):
        print("Processing audio input...")
        audio_data = state["audio_data"]

        # Convert base64 to bytes if needed
        if isinstance(audio_data, str):
            try:
                # Fix common base64 padding issues
                audio_data = audio_data.strip()
                # Add padding if missing
                missing_padding = len(audio_data) % 4
                if missing_padding:
                    audio_data += '=' * (4 - missing_padding)

                audio_data = base64.b64decode(audio_data)
            except Exception as e:
                print(f"Error decoding audio data: {e}")
                print("💡 Tip: Make sure your base64 audio data:")
                print("   - Is a complete base64 string")
                print("   - Has proper padding (= characters at the end)")
                print("   - You can test with: echo 'UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1f' | base64 -d > test.wav")
                return state

        # Transcribe audio to text
        transcribed_text = transcribe_audio(audio_data)

        if transcribed_text:
            print(f"Audio transcribed to: {transcribed_text[:100]}...")
            # Store the transcription for output
            state["audio_transcription"] = transcribed_text
            # Use transcribed text for database querying, but keep original query for reference
            state["user_query"] = transcribed_text
        else:
            print("Failed to transcribe audio, using original query")
            # Keep original user_query as fallback

    elif input_media_type == "image" and state.get("image_data"):
        print("Processing image input...")
        image_data = state["image_data"]

        # Convert base64 to bytes if needed
        if isinstance(image_data, str):
            try:
                # Fix common base64 padding issues
                image_data = image_data.strip()
                # Add padding if missing
                missing_padding = len(image_data) % 4
                if missing_padding:
                    image_data += '=' * (4 - missing_padding)

                image_data = base64.b64decode(image_data)

                # Resize image to reduce token usage
                try:
                    img = Image.open(io.BytesIO(image_data))
                    # Convert to RGB if necessary (for JPEG)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Resize maintaining aspect ratio
                    img.thumbnail((1024, 1024))
                    # Save as JPEG with quality 85
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=85)
                    image_data = buffer.getvalue()
                    print("Image resized to max 1024x1024 for efficiency.")
                except Exception as resize_e:
                    print(f"Warning: Could not resize image: {resize_e}")

            except Exception as e:
                print(f"Error decoding image data: {e}")
                print("💡 Tip: Make sure your base64 image data:")
                print("   - Is a complete base64 string")
                print("   - Has proper padding (= characters at the end)")
                print("   - You can test with: echo 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==' | base64 -d > test.png")
                # Keep original string data for error handling
                image_data = state["image_data"]

        # Get image description for database querying
        description = process_image_input(image_data)

        # Store the description for later use
        state["image_description"] = description
        # Store the transcription for output
        state["image_transcription"] = description

        # Create enhanced query that explains the image context
        enhanced_query = (
            "The user has sent an image. Please analyze and respond based on this image description: "
            f"{description}. "
            "This is the only content provided by the user - no additional text message was included. "
            "Base your response on this image description while considering the context of our conversation."
        )
        state["user_query"] = enhanced_query

        print(f"Image processed, using description for query: {description[:100]}...")

    # For text input or if processing failed, keep the original user_query
    return state


def generate_tts_output(state: State) -> State:
    """
    Generate TTS output based on API-controlled boolean field.

    Args:
        state: Current state with response text

    Returns:
        Updated state with TTS audio data if applicable
    """
    # Check if we should generate TTS based on API field
    should_generate_tts = state.get("should_generate_tts", False)

    # Preserve RAG JSON fields
    rag_fields = {
        'recent_chat_history_json': state.get('recent_chat_history_json', ''),
        'context_data_json': state.get('context_data_json', ''),
        'expert_analysis_json': state.get('expert_analysis_json', ''),
        'interview_and_communication_style_json': state.get('interview_and_communication_style_json', ''),
    }

    if should_generate_tts:
        response_text = state.get("response", "")
        if response_text and response_text.strip():
            print("Generating TTS based on API request")

            # Use the TTS service to generate audio
            updated_state = generate_tts_response(response_text, state)
            return {
                **rag_fields,
                **updated_state
            }

    return {**rag_fields, **state}
