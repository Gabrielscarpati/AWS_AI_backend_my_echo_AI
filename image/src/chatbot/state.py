from typing import List, TypedDict, Dict, Any
from langchain_core.messages import BaseMessage


class State(TypedDict, total=False):
    """Defines the state of the chatbot conversation."""
    user_id: str
    creator_id: str
    influencer_name: str  # Added for dynamic influencer personality
    influencer_personality_prompt: str
    chat_history: List[BaseMessage]
    msgs_cnt_by_user: int
    user_query: str
    retrieved_summaries: str
    conv_response: str
    influencer_answer: str
    influencer_sources: Dict[str, Any]
    response: str
    message_summary: str
    summary_generated: bool
    # Security check fields
    security_check_passed: bool
    security_flags: List[str]
    security_retry_count: int
    original_response: str
    # Media processing fields
    input_media_type: str  # 'text', 'audio', 'image'
    audio_data: bytes  # Base64 encoded audio data
    image_data: bytes  # Base64 encoded image data
    audio_output_url: str  # URL to the generated audio file
    image_description: str  # Description of the image from GPT-4.1-mini
    should_generate_tts: bool  # Whether to generate TTS output (API controlled)
    tts_voice_id: str  # TTS voice ID for this creator
    audio_transcription: str  # User audio transcription from AssemblyAI
    image_transcription: str  # User image transcription from GPT-4.1-nano
    tts_text_sent: str  # Text that was sent to Fish.AI for TTS generation
    # RAG logging fields
    recent_chat_history_json: str
    context_data_json: str
    expert_analysis_json: str
    interview_and_communication_style_json: str
    complete_prompt: str
