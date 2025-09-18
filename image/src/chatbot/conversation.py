import time
import os
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage

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
    ref_lines_all = [f"- [{r['role']}] • {r.get('theme', '')} • {r.get('bullet', '')}" for r in
                     pack.get("reflections", [])]
    mem_lines_all = [f"- ({m.get('source', '')} @ {m.get('created_at', '')}) {m.get('text', '')}" for m in
                     pack.get("memories", [])]
    ref_lines = _truncate_lines(ref_lines_all, max_chars // 2)
    mem_lines = _truncate_lines(mem_lines_all, max_chars // 2)

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
        influencer_name=display_name,
        question=question,
        ref_lines="\n".join(ref_lines) or "- (none)",
        mem_lines="\n".join(mem_lines) or "- (none)",
        conversation_summaries=conversation_summaries,
        recent_chat_history=recent_history_text,
    )
    prefix = (influencer_personality_prompt or "").strip()
    TIMINGS['format_pack'] = time.time() - t0
    return (prefix + "\n\n" + body) if prefix else body


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
    prompt = format_pack(
        creator_id,
        question,
        pack,
        influencer_name=influencer_name,
        conversation_summaries=conversation_summaries,
        influencer_personality_prompt=influencer_personality_prompt,
        recent_chat_history=recent_chat_history,
    )

    provider = provider or ("openai" if os.getenv("OPENAI_API_KEY") else "ollama")
    if provider == "openai":
        model = model or os.getenv("OPENAI_RAG_MODEL", "gpt-4.1-mini")
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
    else:
        text = _ollama_chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
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
    sources = {
        "lenses_used": out.get("lenses_used", []),
        "reflections": [r.get("id") for r in out.get("reflections", [])],
        "memories": [m.get("id") for m in out.get("memories", [])],
    }
    answer_text = out.get("answer", "")
    return {
        "influencer_answer": answer_text,
        "influencer_sources": sources,
        "response": answer_text,
        "timings": TIMINGS.copy(),
    }
