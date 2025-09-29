import json
import time
import os
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import BaseMessage, SystemMessage

from .state import State
from .config import (
    SECURITY_ENABLED, MAX_SECURITY_RETRIES, OPENAI_MODERATION_ENABLED,
    PAST_CHAT_HISTORY_CNT
)
from .conversation import answer_with_rag
from .retrieval import TIMINGS

# Import llm from models module
# from .models import security_llm


def check_openai_moderation(text: str) -> Dict[str, Any]:
    """Check content using OpenAI's moderation API."""
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.moderations.create(input=text)
        result = response.results[0]

        flagged_categories = []
        if result.flagged:
            for category, flagged in result.categories.model_dump().items():
                if flagged:
                    flagged_categories.append(category)

        return {
            "flagged": result.flagged,
            "categories": flagged_categories,
            "scores": result.category_scores.model_dump(),
            "provider": "openai_moderation"
        }
    except Exception as e:
        print(f"OpenAI moderation check failed: {e}")
        return {"flagged": False, "categories": [], "scores": {}, "provider": "openai_moderation", "error": str(e)}


def perform_security_check(text: str) -> Dict[str, Any]:
    """Perform comprehensive security check using multiple methods in parallel."""
    if not SECURITY_ENABLED:
        return {
            "overall_flagged": False,
            "flags": [],
            "checks_performed": [],
            "security_enabled": False
        }

    t0 = time.time()
    checks: List[Dict[str, Any]] = []
    all_flags: List[str] = []

    # Prepare enabled tasks
    tasks = []
    if OPENAI_MODERATION_ENABLED:
        tasks.append(("openai", check_openai_moderation))
    # Removed custom security prompt

    # Execute in parallel
    futures = []
    results_by_name: Dict[str, Dict[str, Any]] = {}
    if tasks:
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            for name, fn in tasks:
                futures.append((name, executor.submit(fn, text)))
            for name, fut in futures:
                try:
                    res = fut.result()
                except Exception as e:
                    # Normalize unexpected errors into a non-flagged result
                    res = {"flagged": False, "categories": [], "error": str(e), "provider": name}
                results_by_name[name] = res
                checks.append(res)
                if res.get("flagged"):
                    cats = res.get("categories", [])
                    all_flags.extend([f"{name}_{cat}" for cat in cats])

    # Determine overall flagged status
    overall_flagged = any(check.get("flagged", False) for check in checks)

    TIMINGS['perform_security_check'] = time.time() - t0

    return {
        "overall_flagged": overall_flagged,
        "flags": all_flags,
        "checks_performed": checks,
        "security_enabled": True,
        "check_duration": TIMINGS['perform_security_check']
    }


def security_check_node(state: State) -> State:
    """Node to perform security checks on the AI response."""
    if not SECURITY_ENABLED:
        return {
            "security_check_passed": True,
            "security_flags": [],
            "security_retry_count": 0
        }

    response_text = state.get("response", "")
    if not response_text:
        return {
            "security_check_passed": True,
            "security_flags": [],
            "security_retry_count": 0
        }

    # Store original response on first check
    if not state.get("original_response"):
        state["original_response"] = response_text

    # Perform security check
    security_result = perform_security_check(response_text)

    if security_result["overall_flagged"]:
        retry_count = state.get("security_retry_count", 0)
        return {
            "security_check_passed": False,
            "security_flags": security_result["flags"],
            "security_retry_count": retry_count,
            "security_check_result": security_result
        }
    else:
        return {
            "security_check_passed": True,
            "security_flags": [],
            "security_retry_count": state.get("security_retry_count", 0),
            "security_check_result": security_result
        }


def regenerate_safe_response(state: State) -> State:
    """Regenerate response with additional safety instructions when flagged."""
    retry_count = state.get("security_retry_count", 0)

    if retry_count >= MAX_SECURITY_RETRIES:
        # Max retries reached, return a safe fallback response
        return {
            "response": "I apologize, but I'm unable to provide a response to that question at this time. Please try rephrasing your question or ask about something else.",
            "security_retry_count": retry_count + 1,
            "security_check_passed": True,
            "security_flags": []
        }

    # Add safety instructions to the generation
    creator_id = state.get('creator_id') or ""
    influencer_name = state.get('influencer_name') or creator_id
    personality = state.get('influencer_personality_prompt', "")
    user_question = state.get('user_query', '')
    conversation_summaries = state.get('retrieved_summaries', '')

    # Get recent chat history
    full_chat_history = state.get('chat_history', [])
    if full_chat_history and getattr(full_chat_history[-1], 'type', '') == 'human':
        recent_chat_history = full_chat_history[:-1][-PAST_CHAT_HISTORY_CNT:]
    else:
        recent_chat_history = full_chat_history[-PAST_CHAT_HISTORY_CNT:]

    # Add safety constraints to personality prompt
    safety_instructions = """
CRITICAL SAFETY REQUIREMENTS:
- Do not share personal information (emails, phone numbers, addresses, etc.)
- Avoid inappropriate, harmful, or offensive content
- Do not provide instructions for illegal activities
- Keep responses appropriate for all audiences
- If unsure about content safety, choose a more conservative response
- Focus on being helpful while maintaining high safety standards
"""

    enhanced_personality = (personality + "\n\n" + safety_instructions).strip()

    tgen = time.time()
    out = answer_with_rag(
        user_question,
        creator_id=creator_id,
        influencer_name=influencer_name,
        conversation_summaries=conversation_summaries,
        influencer_personality_prompt=enhanced_personality,
        recent_chat_history=recent_chat_history,
        temperature=max(0.2, float(os.getenv("INFLUENCER_RAG_TEMPERATURE", 0.4)) - 0.2),  # Lower temperature for safety
        max_tokens=int(os.getenv("INFLUENCER_RAG_MAX_TOKENS", 600)),
        use_cross_encoder=os.getenv("INFLUENCER_RAG_USE_CE", "false").lower() in {"1", "true", "yes", "y"},
    )
    TIMINGS['regenerate_safe_response'] = time.time() - tgen

    return {
        "response": out.get("answer", ""),
        "security_retry_count": retry_count + 1,
        "influencer_answer": out.get("answer", ""),
    }


def should_retry_security(state: State) -> str:
    """Conditional edge function to determine if security check should retry."""
    if not state.get("security_check_passed", True):
        retry_count = state.get("security_retry_count", 0)
        if retry_count < MAX_SECURITY_RETRIES:
            return "regenerate_safe_response"
        else:
            return "summarize"  # Max retries reached, proceed with fallback response
    return "summarize"
