"""
Shared model instances to avoid circular imports.
"""
from langchain.chat_models import init_chat_model

# Initialize LLM once and share across modules (for main chat)
llm = init_chat_model("gpt-4.1-mini", model_provider="openai")

# Initialize separate model for security checks
security_llm = init_chat_model("gpt-4.1-nano", model_provider="openai")
