#!/usr/bin/env python3
"""
Local test script to run the chatbot without Docker.
This simulates the Lambda handler locally with interactive media selection and TTS support.

Usage:
  python local_test.py              # Interactive media selection (text/audio/image)
  python local_test.py --tts       # Interactive media selection with TTS enabled

Features:
- Interactive media type selection (1=text, 2=audio, 3=image)
- Base64 media data input for audio/image
- API-controlled TTS generation
- Full media processing pipeline
- No external dependencies (Supabase-free)

Note: Uses mock test credentials since no external authentication is needed.
"""

import sys
import os
import json
import base64
from pathlib import Path

# Add the image/src directory to Python path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent / "image" / "src"))

# Change to the image/src directory so relative imports work
os.chdir(Path(__file__).parent / "image" / "src")

from app import handler

def get_test_credentials():
    """Mock test credentials since we're not using Supabase."""
    return "test_user_123", "mock_token"

def create_test_event(
    user_message: str,
    user_id: str = None,
    creator_id: str = None,
    influencer_name: str = None,
    influencer_personality_prompt: str = None,
    input_media_type: str = "text",
    should_generate_tts: bool = True,
    media_data: str = None,
):
    """
    Create a test event similar to what Lambda would receive.

    Args:
        user_message: The user's message
        user_id: User ID (optional, defaults to "test_user_123" for testing)
        creator_id: Creator ID (optional, defaults to "aaron_ai")
        influencer_name: Influencer name (optional, defaults to "mihir")
        influencer_personality_prompt: Personality prompt (optional, uses default if not provided)
        input_media_type: Type of input ("text", "audio", "image") - defaults to "text"
        should_generate_tts: Whether to generate TTS output - defaults to False
        media_data: Base64 encoded media data (audio/image) - optional
    """
    if not user_id:
        # Use default test user ID
        user_id = "test_user_123"
    
    # Hardcoded for testing with mihir_ai
    if not creator_id:
        creator_id = "aaron_ai"
    if influencer_name is None:
        influencer_name = "Aaron"
    if influencer_personality_prompt is None:
        influencer_personality_prompt = (
            "Persona: You are Mihir. Speak with intelligence, technical expertise, and thoughtful analysis. "
            "Be helpful, precise, and supportive. Offer specific, practical guidance based on your experiences."
        )

    # Simple chat history with user message - use plain text format
    chat_history = [user_message]
    
    # Prepare the event payload
    payload = {
        "user_id": user_id,
        "creator_id": creator_id,
        "influencer_name": influencer_name,
        "influencer_personality_prompt": influencer_personality_prompt,
        "chat_history": chat_history,
        "msgs_cnt_by_user": 1,
        "input_media_type": input_media_type,
        "user_query": user_message,
        "should_generate_tts": should_generate_tts
    }

    # Add media data if provided
    if media_data and input_media_type in ['audio', 'image']:
        if input_media_type == 'audio':
            payload["audio_data"] = media_data
        elif input_media_type == 'image':
            payload["image_data"] = media_data

    event = {
        "body": json.dumps(payload),
        "isBase64Encoded": False
    }
    
    return event

def select_media_type():
    """Interactive menu to select media input type."""
    print("\n" + "=" * 50)
    print("🎵 MEDIA INPUT SELECTION")
    print("=" * 50)
    print("Choose input type:")
    print("1. Text input (default)")
    print("2. Audio input (base64 encoded)")
    print("3. Image input (base64 encoded)")
    print("-" * 50)

    while True:
        try:
            choice = input("Enter your choice (1-3): ").strip()
            if choice in ['1', '2', '3']:
                break
            print("❌ Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

    media_types = {
        '1': 'text',
        '2': 'audio',
        '3': 'image'
    }

    media_type = media_types[choice]
    media_data = None

    if media_type in ['audio', 'image']:
        print(f"\n📝 Enter base64 encoded {media_type} data:")
        print("(Paste your base64 string and press Enter)")
        print("-" * 50)

        try:
            media_data = input().strip()
            if not media_data:
                print(f"❌ {media_type.capitalize()} data cannot be empty. Using text mode instead.")
                media_type = 'text'
                media_data = None
            else:
                print(f"✅ {media_type.capitalize()} data received ({len(media_data)} characters)")
        except KeyboardInterrupt:
            print(f"\n❌ No {media_type} data provided. Using text mode instead.")
            media_type = 'text'
            media_data = None

    # Show usage examples for the selected media type
    show_media_usage_examples(media_type)

    return media_type, media_data

def show_media_usage_examples(media_type: str):
    """Show examples of how to use the selected media type."""
    print("\n" + "=" * 50)
    print("📋 MEDIA USAGE EXAMPLES")
    print("=" * 50)

    if media_type == 'text':
        print("📝 Text Mode: Just type your message normally")
        print("💬 Example: 'Hello, how are you?'")
    elif media_type == 'audio':
        print("🎵 Audio Mode: Enter base64 encoded audio data")
        print("💡 You can get base64 from:")
        print("   - Recording audio and converting to base64")
        print("   - Speech-to-text tools that output base64")
        print("   - Example: 'UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1f...'")
        print("   - For testing: Use any valid base64 string (system will handle invalid data gracefully)")
    elif media_type == 'image':
        print("🖼️  Image Mode: Enter base64 encoded image data")
        print("💡 You can get base64 from:")
        print("   - Image files converted to base64")
        print("   - Image upload tools that output base64")
        print("   - Use: echo 'data:image/jpeg;base64,'$(base64 image.jpg)")
        print("   - Example: '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCABkAGQDA...'")
        print("   - For testing: Use any valid base64 string (system will handle invalid data gracefully)")

    if media_type in ['audio', 'image']:
        print(f"\n🔄 The {media_type} will be processed and converted to text for analysis")
        print("📝 You can still ask questions about the media in your message")

    print("=" * 50)

def main(enable_tts: bool = False):
    print("🤖 Starting local chatbot test...")
    print("=" * 50)

    # Check if environment variables are set
    required_env_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please update your .env file with the correct values.")
        return

    print("✅ Environment variables loaded")

    # Media type selection
    media_type, media_data = select_media_type()
    print(f"📱 Media mode: {media_type.upper()} {'(with TTS)' if enable_tts else ''}")

    # Interactive mode
    print(f"\nEntering interactive chat mode. Type 'quit' to exit.")
    if media_type == 'text':
        print(f"📱 Current mode: {media_type.upper()}" + (" + TTS" if enable_tts else ""))
    else:
        print(f"📱 Current mode: {media_type.upper()} analysis" + (" + TTS" if enable_tts else ""))
        print("💡 You'll be prompted to add questions about your media")
    print("-" * 50)
    
    user_id = None
    creator_id = "aaron_ai"  # Hardcoded for testing
    influencer_name = "mihir"  # Hardcoded for testing
    influencer_personality_prompt = (
        '''
You are simulating the individual  Aaron.


Your task is to role-play Aaron authentically.  
- Ground every response in these sources.  
- When asked a question, first consider possible responses and align them with Aaron’s traits, values, and past statements.  
- Then provide the response Aaron himself would most likely give.  
- Express answers in Aaron’s natural voice and style (from the transcript), not in an abstract or clinical tone.  
- Do not invent biographical facts or experiences beyond the provided sources.  
- If asked about unfamiliar topics, infer how Aaron would respond using analogies from his known traits, values, and tendencies.  
- When sources conflict, prioritize the interview transcript, then expert analyses, then tests/context.  
- Always remain consistent with Aaron’s character.  


LENGTH
- Every reply MUST be between 1 and 40 words. Rarely exceed 20 words.
Only exceed 20 words if extra detail is needed.
Never exceed 40 words.
- If user requests steps: use ≤5 short bullets, each ≤7 words.


Your knowledge comes only from:
        '''
    )
    
    # Use default test user
    user_id = "test_user_123"
    print(f"✅ Using test user: {user_id[:8]}...")

    msg_count = 0
    chat_history = []  # accumulate plain text messages across turns
    
    while True:
        try:
            # Handle input based on media type
            user_input = ""
            if media_type == 'text':
                user_input = input("\n💬 You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue
            elif media_type in ['audio', 'image']:
                # For audio/image, ask if they want to add a text question
                print(f"\n📱 {media_type.upper()} mode active")
                print("💬 Add a question about your media? (press Enter to skip)")
                text_question = input("🔍 Question: ").strip()

                if text_question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                user_input = text_question if text_question else f"Please analyze this {media_type}"

                # Show what will be sent
                if text_question:
                    print(f"✅ Sending: '{user_input}' with {media_type} data")
                else:
                    print(f"✅ Sending analysis request with {media_type} data")
            else:
                print("❌ Invalid media type")
                break

            msg_count += 1

            # Append user's message to the running chat history as plain text
            chat_history.append(user_input)

            # Create test event and override with accumulated history
            event = create_test_event(
                user_input,
                user_id,
                creator_id,
                influencer_name,
                influencer_personality_prompt,
                input_media_type=media_type,
                should_generate_tts=enable_tts,
                media_data=media_data,
            )
            event["body"] = json.loads(event["body"])  # to dict for mutation
            event["body"]["msgs_cnt_by_user"] = msg_count
            event["body"]["chat_history"] = chat_history
            event["body"] = json.dumps(event["body"])  # back to json string
            
            print("🤔 Haven is thinking...")
            
            # Call the handler
            response = handler(event, {})
            
            if response["statusCode"] == 200:
                response_data = json.loads(response["body"])
                print(f"🤖 Haven: {response_data['response']}")
                # Append assistant message to the running chat history as plain text
                chat_history.append(response_data.get('response', ''))

                if response_data.get('audio_output'):
                    print("🔊 Audio response generated (base64 encoded)")
                    print(f"   Audio URL available: {response_data.get('audio_output_url', 'N/A')[:50]}...")
                    print(f"   Audio size: {len(response_data['audio_output'])} bytes")

                    audio_file = f"local_test_audio_msg_{msg_count}.mp3"
                    try:
                        audio_bytes = base64.b64decode(response_data['audio_output'])
                        with open(audio_file, "wb") as f:
                            f.write(audio_bytes)
                        print(f"   💾 Saved to: {audio_file}")
                    except Exception as e:
                        print(f"   ❌ Failed to save audio: {e}")

                # Show media processing info
                input_media_type = response_data.get('input_media_type', 'text')
                if input_media_type != 'text':
                    image_description = response_data.get('image_description', '')
                    if image_description:
                        print(f"🖼️ Image described as: {image_description[:100]}...")

                timings = response_data.get('timings', {})
                if timings:
                    # Calculate total avoiding double-counting nested timings
                    # generate_influencer_answer includes influencer_retrieve, format_pack, and answer_with_rag_model_call
                    # So we only count the top-level operations: retrieve_context + generate_influencer_answer
                    top_level_timings = ['retrieve_context', 'generate_influencer_answer']
                    total = sum(timings.get(k, 0) for k in top_level_timings if k in timings)
                    
                    # Print individual timings and corrected total
                    for k, v in timings.items():
                        print(f"   ⏱ {k}: {v:.3f}s")
                    print(f"   ⏱ total: {total:.3f}s")
            else:
                error_data = json.loads(response["body"])
                print(f"❌ Error: {error_data.get('error', 'Unknown error')}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please check your configuration and try again.")

    # Show summary of generated audio files
    print("\n" + "=" * 50)
    print("🎵 AUDIO GENERATION SUMMARY")
    print("=" * 50)

    audio_files = [f for f in os.listdir('.') if f.startswith('local_test_audio_msg_') and f.endswith('.mp3')]
    if audio_files:
        print(f"✅ Generated {len(audio_files)} audio files:")
        for i, audio_file in enumerate(audio_files, 1):
            file_size = os.path.getsize(audio_file)
            print(f"   {i}. {audio_file} ({file_size:,} bytes)")

        print(f"\n💡 You can listen to any audio file:")
        print(f"   open {audio_files[0]}  # Play the first one")
    else:
        print("ℹ️  No audio files were generated in this session.")

    print("=" * 50)

if __name__ == "__main__":
    import sys
    # Check for TTS flag in command line arguments
    enable_tts = "--tts" in sys.argv
    if enable_tts:
        print("🎵 TTS generation enabled for all messages")
    main(enable_tts)