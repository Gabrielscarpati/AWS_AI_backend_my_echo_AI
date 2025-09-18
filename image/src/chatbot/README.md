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

## Architecture

The system uses LangGraph to orchestrate the following flow:

1. **retrieve_context** - Fetch relevant conversation summaries
2. **generate_influencer_answer** - Generate response using RAG
3. **security_check_node** - Check response for safety
4. **regenerate_safe_response** - Retry with safety constraints if needed
5. **summarize** - Generate conversation summary when threshold reached

## Benefits of This Structure

- **Modularity**: Each file has a single responsibility
- **Maintainability**: Easier to find and modify specific functionality
- **Testability**: Individual modules can be tested in isolation
- **Scalability**: New features can be added as separate modules
- **Import Safety**: Circular imports are avoided through careful dependency management
