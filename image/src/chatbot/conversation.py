import time
import os
import base64
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage
import json

from .state import State
from .config import PAST_CHAT_HISTORY_CNT
from .retrieval import prompt_templates, TIMINGS
from .influencer_rag import influencer_retrieve
from .utils import _truncate_lines
from .models import llm


def generate_conversation_response(state: State) -> State:
    """Generate response using conversation model."""
    system_prompt = prompt_templates['MAIN_SYSTEM_PROMPT']
    system_prompt = system_prompt.format(summaries=state['retrieved_summaries'])
    # Send only the latest configured messages to the conversation model for efficiency
    history_to_send = state.get('chat_history', [])[-PAST_CHAT_HISTORY_CNT:]
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *history_to_send
    ])
    return {"conv_response": str(response.content)}


def format_pack(
        creator_id: str,
        question: str,
        pack: Dict[str, Any],
        influencer_name: str = None,
        conversation_summaries: str = "",
        influencer_personality_prompt: str = "",
        recent_chat_history: List[BaseMessage] = None,
        max_chars: int = 6000,
) -> str:
    """Format the retrieval pack into a prompt."""
    # Do not include internal ids in the prompt lines
    # New categories
    ctx_lines_all = [f"- {c.get('source', '')} • {c.get('text', '')}" for c in pack.get("context_data", [])]
    exp_lines_all = [f"- {e.get('title', '')}: {e.get('text', '')}" for e in pack.get("expert_analysis", [])]
    ivw_lines_all = [f"- {i.get('model_type', '')} • {i.get('sub_category', '')}: {i.get('full_text', i.get('text', ''))}" for i in pack.get("interview_styles", [])]
    # Allocate space across the three sections
    ctx_lines = _truncate_lines(ctx_lines_all, max_chars // 3)
    exp_lines = _truncate_lines(exp_lines_all, max_chars // 3)
    ivw_lines = _truncate_lines(ivw_lines_all, max_chars // 3)

    # Use influencer name if provided, otherwise fallback to creator_id
    display_name = influencer_name or creator_id

    # Format recent chat history
    recent_history_text = ""
    if recent_chat_history:
        history_lines = []
        for msg in recent_chat_history:
            role = "USER" if msg.type == "human" else "ASSISTANT"
            history_lines.append(f"{role}: {msg.content}")
        recent_history_text = "\n".join(history_lines)
    
    if not recent_history_text:
        recent_history_text = "(no recent chat history)"

    # Use the dynamic template from YAML
    template = prompt_templates['DYNAMIC_INFLUENCER_PROMPT']
    t0 = time.time()
    body = template.format(
        interview_and_communication_style="\n".join(ivw_lines) or "- (none)",
        expert_analysis="\n".join(exp_lines) or "- (none)",
        context_data="\n".join(ctx_lines) or "- (none)",
        question=question,
        conversation_summaries=conversation_summaries,
        recent_chat_history=recent_history_text,
    )
    prefix = (influencer_personality_prompt or "").strip()
    TIMINGS['format_pack'] = time.time() - t0
    return (prefix + "\n\n" + body) if prefix else body


def _mistral_chat(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI
    import os
    client = OpenAI(
        base_url="https://api.mistral.ai/v1",
        api_key=os.getenv("MISTRAL_API_KEY")
    )
    res = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return res.choices[0].message.content


def _openai_chat(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
    """Call OpenAI chat API."""
    from openai import OpenAI
    client = OpenAI()
    res = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return res.choices[0].message.content


def _ollama_chat(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
    """Call Ollama chat API."""
    import requests
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    url = f"{base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def answer_with_rag(
        question: str,
        creator_id: str,
        influencer_name: str = None,
        conversation_summaries: str = "",
        influencer_personality_prompt: str = "",
        recent_chat_history: List[BaseMessage] = None,
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 600,
        use_cross_encoder: bool = True,
) -> Dict[str, Any]:
    """Answer question using RAG approach."""
    pack = influencer_retrieve(question, creator_id=creator_id, use_cross_encoder=use_cross_encoder)
    
    # Debug: Print pack structure
    print(f"DEBUG pack keys: {list(pack.keys())}")
    print(f"DEBUG pack structure: { {k: type(v).__name__ + f'({len(v)})' if isinstance(v, list) else type(v).__name__ for k, v in pack.items()} }")
    
    prompt = format_pack(
        creator_id,
        question,
        pack,
        influencer_name=influencer_name,
        conversation_summaries=conversation_summaries,
        influencer_personality_prompt=influencer_personality_prompt,
        recent_chat_history=recent_chat_history,
    )

    provider = provider or ("mistral" if os.getenv("MISTRAL_API_KEY") else "openai" if os.getenv("OPENAI_API_KEY") else "ollama")
    if provider == "openai":
        model = model or os.getenv("OPENAI_RAG_MODEL", "gpt-4.1-nano")
    elif provider == "mistral":
        model = model or "mistral-small-2506"
    else:
        model = model or os.getenv("OLLAMA_RAG_MODEL", "llama3.1")

    messages = [
        {"role": "system",
         "content": "You are a helpful assistant who speaks in the influencer's authentic voice while staying factual."},
        {"role": "user", "content": prompt}
    ]

    tmodel = time.time()
    if provider == "openai":
        text = _openai_chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == "mistral":
        text = _mistral_chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        text = _ollama_chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    TIMINGS['answer_with_rag_model_call'] = time.time() - tmodel
    
    # Debug: Print extracted RAG data before return
    context_data = pack.get("context_data", [])
    expert_analysis = pack.get("expert_analysis", [])
    interview_styles = pack.get("interview_styles", [])
    print(f"DEBUG extracted RAG: context={len(context_data)}, expert={len(expert_analysis)}, interview={len(interview_styles)}")
    if context_data:
        print(f"DEBUG sample context: {context_data[0]}")
    if expert_analysis:
        print(f"DEBUG sample expert: {expert_analysis[0]}")
    if interview_styles:
        print(f"DEBUG sample interview: {interview_styles[0]}")

    return {
        "provider": provider,
        "model": model,
        "question": question,
        "lenses_used": pack.get("lenses_used", []),
        "reflections": pack.get("reflections", []),
        "memories": pack.get("memories", []),
        "answer": text,
        "context_data": context_data,
        "expert_analysis": expert_analysis,
        "interview_and_communication_style": interview_styles
    }


def answer_with_rag_and_image(
        question: str,
        image_data: bytes,
        creator_id: str,
        influencer_name: str = None,
        conversation_summaries: str = "",
        influencer_personality_prompt: str = "",
        recent_chat_history: List[BaseMessage] = None,
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 600,
        use_cross_encoder: bool = True,
) -> Dict[str, Any]:
    """Answer question using RAG approach with image input."""
    import base64

    pack = influencer_retrieve(question, creator_id=creator_id, use_cross_encoder=use_cross_encoder)
    prompt = format_pack(
        creator_id,
        question,
        pack,
        influencer_name=influencer_name,
        conversation_summaries=conversation_summaries,
        influencer_personality_prompt=influencer_personality_prompt,
        recent_chat_history=recent_chat_history,
    )

    image_base64 = base64.b64encode(image_data).decode('utf-8')

    provider = provider or ("mistral" if os.getenv("MISTRAL_API_KEY") else "openai" if os.getenv("OPENAI_API_KEY") else "ollama")
    if provider == "openai":
        model = model or os.getenv("OPENAI_RAG_MODEL", "gpt-4.1-nano")
    elif provider == "mistral":
        model = model or "mistral-small-latest"
    else:
        model = model or os.getenv("OLLAMA_RAG_MODEL", "llama3.1")

    tmodel = time.time()
    if provider == "openai":
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant who speaks in the influencer's authentic voice while staying factual. You can see and analyze images provided by the user."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}
        ]
        text = _openai_chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == "mistral":
        # Mistral doesn't support vision yet, fallback to text-only
        messages_fallback = [
            {"role": "system",
             "content": "You are a helpful assistant who speaks in the influencer's authentic voice while staying factual. The user has provided an image related to their question, but image analysis is not available."},
            {"role": "user", "content": prompt}
        ]
        text = _mistral_chat(messages_fallback, model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        messages_fallback = [
            {"role": "system",
             "content": "You are a helpful assistant who speaks in the influencer's authentic voice while staying factual. The user has provided an image, but I can only describe it as: " + question},
            {"role": "user", "content": prompt}
        ]
        text = _ollama_chat(messages_fallback, model=model, temperature=temperature, max_tokens=max_tokens)
    TIMINGS['answer_with_rag_model_call'] = time.time() - tmodel

    return {
        "provider": provider,
        "model": model,
        "question": question,
        "lenses_used": pack.get("lenses_used", []),
        "reflections": pack.get("reflections", []),
        "memories": pack.get("memories", []),
        "answer": text
    }


def generate_influencer_answer(state: State) -> State:
    """Generate influencer answer using RAG."""
    creator_id = state.get('creator_id') or ""
    # Get influencer name from state or use creator_id as fallback
    influencer_name = state.get('influencer_name') or creator_id
    personality = state.get('influencer_personality_prompt', "")

    # Pass conversation summaries separately rather than combining with question
    user_question = state.get('user_query', '')
    conversation_summaries = state.get('retrieved_summaries', '')

    # Get recent chat history excluding the current user message to avoid duplication with User question
    full_chat_history = state.get('chat_history', [])
    if full_chat_history and getattr(full_chat_history[-1], 'type', '') == 'human':
        recent_chat_history = full_chat_history[:-1][-PAST_CHAT_HISTORY_CNT:]
    else:
        recent_chat_history = full_chat_history[-PAST_CHAT_HISTORY_CNT:]

    # For images, user_query already includes the description from media_processing
    # No need for special image handling here - use text-only RAG

    tgen = time.time()

    out = answer_with_rag(
        user_question,
        creator_id=creator_id,
        influencer_name=influencer_name,
        conversation_summaries=conversation_summaries,
        influencer_personality_prompt=personality,
        recent_chat_history=recent_chat_history,
        temperature=float(os.getenv("INFLUENCER_RAG_TEMPERATURE", 0.4)),
        max_tokens=int(os.getenv("INFLUENCER_RAG_MAX_TOKENS", 600)),
        use_cross_encoder=os.getenv("INFLUENCER_RAG_USE_CE", "false").lower() in {"1", "true", "yes", "y"},
    )

    TIMINGS['generate_influencer_answer'] = time.time() - tgen
    
    # Debug: Print RAG data lengths to verify retrieval
    print(f"DEBUG RAG data counts: context={len(out.get('context_data', []))}, expert={len(out.get('expert_analysis', []))}, interview={len(out.get('interview_and_communication_style', []))}")
    
    sources = {
        "lenses_used": out.get("lenses_used", []),
        "context_data": [r.get("id") for r in out.get("context_data", [])],
        "expert_analysis": [m.get("id") for m in out.get("expert_analysis", [])],
        "interview_styles": [m.get("id") for m in out.get("interview_and_communication_style", [])],
    }
    answer_text = out.get("answer", "")
    recent_history_json = json.dumps([{"role": "USER" if msg.type == "human" else "ASSISTANT", "content": msg.content} for msg in recent_chat_history])
    context_data_json = json.dumps(out.get("context_data", []))
    expert_analysis_json = json.dumps(out.get("expert_analysis", []))
    interview_style_json = json.dumps(out.get("interview_and_communication_style", []))
    return {
        "influencer_answer": answer_text,
        "influencer_sources": sources,
        "response": answer_text,
        "timings": TIMINGS.copy(),
        "recent_chat_history_json": recent_history_json,
        "context_data_json": context_data_json,
        "expert_analysis_json": expert_analysis_json,
        "interview_and_communication_style_json": interview_style_json,
    }
