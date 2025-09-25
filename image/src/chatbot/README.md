# Chatbot Module Structure

This folder contains the reorganized chatbot logic, split from the original `chatbot_clio.py` into focused, maintainable modules.

## Module Overview

### Core Files

- **`orchestrator.py`** - Main entry point that coordinates all modules using LangGraph
- **`state.py`** - Defines the conversation state structure
- **`config.py`** - All configuration constants and environment variables
- **`models.py`** - Shared model instances (LLM) to avoid circular imports

### Functional Modules

- **`retrieval.py`** - Context retrieval from vector database
- **`influencer_rag.py`** - Influencer-specific RAG retrieval and processing
- **`conversation.py`** - Conversation generation and RAG answer formatting
- **`security.py`** - Content moderation and security checks
- **`summarization.py`** - Conversation summarization logic
- **`utils.py`** - Shared utility functions

### Resources

- **`prompt_templates.yaml`** - All prompt templates used by the system

## Usage

The main entry point is `orchestrator.py` which exports `chatbot_clio` - the compiled LangGraph workflow.

```python
from chatbot.orchestrator import chatbot_clio, PATTERN_USER

# Use chatbot_clio.invoke(state) as before
result = chatbot_clio.invoke(state)
```

### Media Input Usage

To use media inputs, include these fields in your request payload:

```json
{
  "user_id": "user123",
  "creator_id": "influencer123",
  "influencer_name": "influencer_name",
  "influencer_personality_prompt": "optional personality",
  "chat_history": [],
  "msgs_cnt_by_user": 1,
  "input_media_type": "audio",  // "text", "audio", or "image"
  "audio_data": "base64_encoded_audio_data",  // Required for audio input
  "image_data": "base64_encoded_image_data",  // Required for image input
  "user_query": "text query",  // For text input or as backup
  "should_generate_tts": false  // Boolean to control TTS generation
}
```

### Environment Variables

Required environment variables for media functionality:

- `ELEVENLABS_API_KEY` - Your ElevenLabs API key for TTS
- `ELEVENLABS_VOICE_ID` - Voice ID to use (optional, defaults to 21m00Tcm4TlvDq8ikWAM)
- `TTS_PROVIDER` - "elevenlabs" or "f5tts" (optional, defaults to "elevenlabs")

All existing environment variables are still required:
- `OPENAI_API_KEY` - For GPT models and Whisper STT
- `PINECONE_API_KEY` - For vector database

## Media Support

The chatbot now supports audio and image inputs in addition to text:

### Input Types

- **Text Input** (default): Regular text messages
- **Audio Input**: Speech-to-text conversion using OpenAI Whisper
- **Image Input**: Image description using GPT-4.1-mini, then processed as text

### Audio Output (TTS)

Every 15th user message automatically triggers text-to-speech conversion using ElevenLabs API.

### Media Processing Flow

1. **process_media_input** - Handle audio/image to text conversion
2. **retrieve_context** - Fetch relevant conversation summaries
3. **generate_influencer_answer** - Generate response using RAG (with image support)
4. **security_check_node** - Check response for safety
5. **regenerate_safe_response** - Retry with safety constraints if needed
6. **summarize** - Generate conversation summary when threshold reached
7. **generate_tts_output** - Generate audio output for every 15th message

### New Modules

- **`media_processing.py`** - Handles audio/image input processing
- **`stt_service.py`** - OpenAI Whisper speech-to-text service
- **`image_service.py`** - GPT-4.1-mini image description service
- **`tts_service.py`** - ElevenLabs text-to-speech service

## Architecture

The system uses LangGraph to orchestrate the following flow:

## Benefits of This Structure

- **Modularity**: Each file has a single responsibility
- **Maintainability**: Easier to find and modify specific functionality
- **Testability**: Individual modules can be tested in isolation
- **Scalability**: New features can be added as separate modules
- **Import Safety**: Circular imports are avoided through careful dependency management
