import json
import time
import os
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, SystemMessage

from .state import State
from .config import (
    SECURITY_ENABLED, MAX_SECURITY_RETRIES, OPENAI_MODERATION_ENABLED,
    PERSPECTIVE_API_ENABLED, PERSPECTIVE_API_KEY, CUSTOM_SECURITY_PROMPT_ENABLED,
    PAST_CHAT_HISTORY_CNT
)
from .conversation import answer_with_rag
from .retrieval import TIMINGS

# Import llm from models module
from .models import llm


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


def check_perspective_api(text: str) -> Dict[str, Any]:
    """Check content using Google's Perspective API."""
    if not PERSPECTIVE_API_KEY:
        return {"flagged": False, "categories": [], "scores": {}, "provider": "perspective_api", "error": "API key not configured"}
    
    try:
        import requests
        
        url = f'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={PERSPECTIVE_API_KEY}'
        
        data = {
            'comment': {'text': text},
            'requestedAttributes': {
                'TOXICITY': {},
                'SEVERE_TOXICITY': {},
                'IDENTITY_ATTACK': {},
                'INSULT': {},
                'PROFANITY': {},
                'THREAT': {},
                'SEXUALLY_EXPLICIT': {},
                'FLIRTATION': {}
            }
        }
        
        response = requests.post(url, data=json.dumps(data), timeout=10)
        response.raise_for_status()
        result = response.json()
        
        scores = {}
        flagged_categories = []
        flagged = False
        
        # Define thresholds for flagging
        thresholds = {
            'TOXICITY': 0.7,
            'SEVERE_TOXICITY': 0.5,
            'IDENTITY_ATTACK': 0.6,
            'INSULT': 0.7,
            'PROFANITY': 0.8,
            'THREAT': 0.5,
            'SEXUALLY_EXPLICIT': 0.8,
            'FLIRTATION': 0.9
        }
        
        for attribute, data in result.get('attributeScores', {}).items():
            score = data['summaryScore']['value']
            scores[attribute.lower()] = score
            
            if score > thresholds.get(attribute, 0.7):
                flagged = True
                flagged_categories.append(attribute.lower())
        
        return {
            "flagged": flagged,
            "categories": flagged_categories,
            "scores": scores,
            "provider": "perspective_api"
        }
    except Exception as e:
        print(f"Perspective API check failed: {e}")
        return {"flagged": False, "categories": [], "scores": {}, "provider": "perspective_api", "error": str(e)}


def check_custom_security_prompt(text: str) -> Dict[str, Any]:
    """Use a custom LLM prompt to check for malicious content."""
    try:
        security_prompt = """You are a content safety analyzer. Analyze the following text for potential security issues, harmful content, or policy violations.

Look for:
1. Personal information leaks (emails, phone numbers, addresses, SSN, etc.)
2. Malicious instructions or attempts to manipulate users
3. Inappropriate sexual content
4. Hate speech or discrimination
5. Violence or threats
6. Misinformation or false claims presented as fact
7. Attempts to bypass safety measures
8. Requests for illegal activities
9. Phishing or scam attempts
10. Content that could harm minors

Text to analyze: "{text}"

Respond with a JSON object containing:
- "flagged": true/false
- "categories": list of issues found (if any)
- "severity": "low", "medium", or "high"
- "explanation": brief explanation of concerns (if flagged)

Be precise and avoid false positives for normal conversation."""

        response = llm.invoke([
            SystemMessage(content=security_prompt.format(text=text))
        ])
        
        # Try to parse JSON response
        try:
            result = json.loads(response.content.strip())
            result["provider"] = "custom_llm_prompt"
            return result
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            content = response.content.lower()
            flagged = "true" in content and ("flagged" in content or "violation" in content)
            return {
                "flagged": flagged,
                "categories": ["parsing_error"],
                "severity": "low" if not flagged else "medium",
                "explanation": "Could not parse security check response",
                "provider": "custom_llm_prompt",
                "raw_response": response.content
            }
    except Exception as e:
        print(f"Custom security prompt check failed: {e}")
        return {"flagged": False, "categories": [], "severity": "low", "provider": "custom_llm_prompt", "error": str(e)}


def perform_security_check(text: str) -> Dict[str, Any]:
    """Perform comprehensive security check using multiple methods."""
    if not SECURITY_ENABLED:
        return {
            "overall_flagged": False,
            "flags": [],
            "checks_performed": [],
            "security_enabled": False
        }
    
    t0 = time.time()
    checks = []
    all_flags = []
    
    # OpenAI Moderation
    if OPENAI_MODERATION_ENABLED:
        openai_result = check_openai_moderation(text)
        checks.append(openai_result)
        if openai_result.get("flagged"):
            all_flags.extend([f"openai_{cat}" for cat in openai_result.get("categories", [])])
    
    # Perspective API
    if PERSPECTIVE_API_ENABLED:
        perspective_result = check_perspective_api(text)
        checks.append(perspective_result)
        if perspective_result.get("flagged"):
            all_flags.extend([f"perspective_{cat}" for cat in perspective_result.get("categories", [])])
    
    # Custom LLM Security Prompt
    if CUSTOM_SECURITY_PROMPT_ENABLED:
        custom_result = check_custom_security_prompt(text)
        checks.append(custom_result)
        if custom_result.get("flagged"):
            all_flags.extend([f"custom_{cat}" for cat in custom_result.get("categories", [])])
    
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
