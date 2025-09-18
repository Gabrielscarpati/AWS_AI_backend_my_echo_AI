from dotenv import load_dotenv

load_dotenv()

import yaml
import uuid
import re
import os
import time

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from typing import List, TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from supabase_utils import get_messages, create_message, get_total_messages_cnt_by_user, get_test_credentials

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pprint import pprint

# ---- Influencer RAG imports (merged from newfeatureschat.py) ----
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from functools import lru_cache
import httpx
import requests

_CE = None


def _maybe_load_cross_encoder():
    global _CE
    if _CE is not None:
        return _CE
    try:
        from sentence_transformers import CrossEncoder
        _CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        _CE = None
    return _CE


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
INDEX_DIR = Path(os.getenv("INFLUENCER_INDEX_DIR", "indexes"))
MEM_FILE = INDEX_DIR / "memories.json"
REF_FILE = INDEX_DIR / "reflections.json"

# Timing measurements for instrumentation
TIMINGS: Dict[str, float] = {}

with open("prompt_templates.yaml") as prompt_template_file:
    prompt_templates = yaml.safe_load(prompt_template_file)

flags = re.DOTALL | re.MULTILINE
PATTERN_USER = re.compile(r"<user>(.*?)<\/user>", flags=flags)
PATTERN_ASSISTANT = re.compile(r"<assistant>(.*?)<\/assistant>", flags=flags)

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

RETRIEVAL_SUMMARY_CNT = 3
CONVERSATION_SUMMARY_THRESHOLD = 9
# Number of recent chat messages to send to models (configurable via env)
PAST_CHAT_HISTORY_CNT = 9
EMBEDDING_DIMENSION = 1024

# Security configuration
SECURITY_ENABLED = os.getenv("SECURITY_ENABLED", "true").lower() in {"1", "true", "yes", "y"}
MAX_SECURITY_RETRIES = int(os.getenv("MAX_SECURITY_RETRIES", "2"))
OPENAI_MODERATION_ENABLED = os.getenv("OPENAI_MODERATION_ENABLED", "true").lower() in {"1", "true", "yes", "y"}
PERSPECTIVE_API_ENABLED = os.getenv("PERSPECTIVE_API_ENABLED", "false").lower() in {"1", "true", "yes", "y"}
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")
CUSTOM_SECURITY_PROMPT_ENABLED = os.getenv("CUSTOM_SECURITY_PROMPT_ENABLED", "true").lower() in {"1", "true", "yes", "y"}

llm = init_chat_model("gpt-4.1-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=EMBEDDING_DIMENSION)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("influencer-conversation-summary")
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

try:
    influencer_index = pc.Index("influencer-brain")
except Exception:
    influencer_index = None


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


def messages_to_txt(messages: List[BaseMessage]) -> str:
    role_map = {
        'human': 'user',
        'ai': 'assistant',
    }
    return "\n".join([
        f"{role_map[message.type].upper()}: {message.content}"
        for message in messages
    ])


def retrieve_context(state: State) -> State:
    """Retrieves relevant summaries of conversation from the vector store"""
    # Reset timings for each invocation and use raw last user message as the query if not provided
    TIMINGS.clear()
    t0 = time.time()
    query = state.get("user_query")
    if not query:
        chat_history = state.get("chat_history", [])
        query = chat_history[-1].content if chat_history else ""

    retrieved_docs = vector_store.similarity_search(
        query,
        k=RETRIEVAL_SUMMARY_CNT,
        filter={"user_id": state["user_id"]}
    )

    TIMINGS['retrieve_context'] = time.time() - t0

    retrieved_summaries = "\n\n".join([
        f"SUMMARY {i}:\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs, start=1)
    ])
    return {"retrieved_summaries": retrieved_summaries, "user_query": query}


# -------------------------
# Influencer RAG helpers (retrieve + prompt assembly + generation)
# -------------------------

_model_st: SentenceTransformer = SentenceTransformer(EMBEDDING_MODEL)


def _get_st_model() -> SentenceTransformer:
    return _model_st


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _texts(rows: List[Dict[str, Any]]) -> List[str]:
    out = []
    for r in rows:
        out.append(r["bullet"] if r.get("type") == "reflection" else r.get("text", ""))
    return out


def _emb_matrix(rows: List[Dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 384), dtype=np.float32)
    arr = np.array([r.get("embedding", np.zeros(384, dtype=np.float32)) for r in rows], dtype=np.float32)
    return arr


def _embed_query(q: str) -> np.ndarray:
    return _get_st_model().encode([q], normalize_embeddings=True)[0]


def _topk_dense(qv: np.ndarray, vecs: np.ndarray, k: int) -> List[int]:
    if vecs.shape[0] == 0 or k <= 0:
        return []
    sims = vecs @ qv
    idx = np.argsort(sims)[-k:][::-1]
    return idx.tolist()


def _topk_bm25(bm25: BM25Okapi, query: str, k: int) -> List[int]:
    if bm25 is None or k <= 0:
        return []
    toks = query.split()
    scores = bm25.get_scores(toks)
    idx = np.argsort(scores)[-k:][::-1]
    return idx.tolist()


def _mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, lambda_mult=0.7, k=10) -> List[int]:
    if cand_vecs.shape[0] == 0:
        return []
    remaining = list(range(cand_vecs.shape[0]))
    selected: List[int] = []
    sim_q = cand_vecs @ query_vec
    while remaining and len(selected) < k:
        if not selected:
            best = int(np.argmax(sim_q[remaining]))
            selected.append(remaining.pop(best))
            continue
        mmr_scores = []
        for j in remaining:
            redundancy = np.max(cand_vecs[j] @ cand_vecs[selected].T)
            mmr_scores.append(lambda_mult * sim_q[j] - (1 - lambda_mult) * redundancy)
        j_local = int(np.argmax(mmr_scores))
        selected.append(remaining.pop(j_local))
    return selected


def _dedupe_keep_order(items: List[int]) -> List[int]:
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _dedupe_rows_keep_order(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for r in rows:
        key = f"{r.get('type', '?')}::{r.get('id', '')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _cross_encoder_rerank(query: str, pool_rows: List[Dict[str, Any]]) -> List[int]:
    CE = _maybe_load_cross_encoder()
    if CE is None or not pool_rows:
        return list(range(len(pool_rows)))
    texts = []
    for r in pool_rows:
        txt = r["bullet"] if r.get("type") == "reflection" else r.get("text", "")
        texts.append(txt)
    pairs = [[query, t] for t in texts]
    scores = CE.predict(pairs)
    order = np.argsort(scores)[::-1].tolist()
    return order


LENS_KEYWORDS = {
    "behav_econ": ["price", "budget", "cpm", "roi", "usage", "rights", "whitelist", "format", "deliverable", "contract",
                   "payment", "sponsorship", "accept", "deal"],
    "psych": ["tone", "voice", "style", "boundaries", "values", "ethics", "creative", "script", "tone of voice", "dm",
              "intro"],
    "political": ["politics", "controversial", "avoid", "red line", "endorsement", "mlm", "diet"],
    "demo": ["audience", "demographic", "age", "country", "region", "when", "time", "timezone", "peak", "engagement"],
}
ORDER = ["behav_econ", "psych", "demo", "political"]


def _pick_lenses(q: str, top: int = 2) -> List[str]:
    ql = q.lower()
    scores = {k: 0 for k in LENS_KEYWORDS}
    for lens, kws in LENS_KEYWORDS.items():
        for kw in kws:
            if kw in ql:
                scores[lens] += 1
    hit = [k for k, v in scores.items() if v > 0]
    lenses = hit if hit else ORDER
    return lenses[:top]


def influencer_retrieve(query: str, creator_id: str, top_ref: int = 5, top_mem: int = 8,
                        use_cross_encoder: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    lenses = _pick_lenses(query)
    qv = _embed_query(query)

    def _from_pinecone() -> Dict[str, Any] | None:
        if influencer_index is None:
            return None
        try:
            # Single query with OR filter; fetch a wider pool then partition locally
            ref_clause: Dict[str, Any] = {"type": "reflection"}
            if lenses:
                ref_clause["role"] = {"$in": lenses}

            combined_filter: Dict[str, Any] = {
                "creator_id": creator_id,
                "$or": [
                    ref_clause,
                    {"type": "memory"}
                ]
            }

            # BOTTLENECK #1 FIX: Reduce top_k and remove vectors from payload
            pinecone_top_k = min(24, max(top_ref * 3 + top_mem * 3, 16))
            all_res = influencer_index.query(
                vector=qv.tolist(),
                top_k=pinecone_top_k,
                filter=combined_filter,
                include_metadata=True,
                include_values=False,  # <-- BIG CHANGE
            )

            matches = getattr(all_res, "matches", None) or all_res.get("matches", [])
            ref_rows: List[Dict[str, Any]] = []
            mem_rows: List[Dict[str, Any]] = []

            for m in matches:
                md = getattr(m, "metadata", None) or m.get("metadata", {})
                rid = getattr(m, "id", None) or m.get("id")
                # vals = getattr(m, "values", None) or m.get("values") # No longer fetching values
                mtype = md.get("type")
                if mtype == "reflection":
                    ref_rows.append({
                        "type": "reflection",
                        "id": rid,
                        "creator_id": md.get("creator_id"),
                        "role": md.get("role"),
                        "theme": md.get("theme"),
                        "bullet": md.get("bullet", ""),
                        "source_ids": md.get("source_ids", []),
                        "created_at": md.get("created_at"),
                        # "embedding": vals, # No longer fetching values
                    })
                elif mtype == "memory":
                    mem_rows.append({
                        "type": "memory",
                        "id": rid,
                        "creator_id": md.get("creator_id"),
                        "text": md.get("text", ""),
                        "source": md.get("source"),
                        "platform": md.get("platform"),
                        "url": md.get("url"),
                        "created_at": md.get("created_at"),
                        "topics": md.get("topics", []),
                        "privacy_level": md.get("privacy_level"),
                        # "embedding": vals, # No longer fetching values
                    })

            return {"ref_rows": ref_rows, "mem_rows": mem_rows}
        except Exception:
            return None

    def _from_json() -> Dict[str, Any]:
        all_refs = [r for r in _load_rows(REF_FILE) if r.get("creator_id") == creator_id]
        all_mems = [m for m in _load_rows(MEM_FILE) if m.get("creator_id") == creator_id]
        ref_rows = [r for r in all_refs if r.get("role") in lenses] or all_refs
        mem_rows = all_mems
        return {"ref_rows": ref_rows, "mem_rows": mem_rows}

    fetched = _from_pinecone() or _from_json()
    ref_rows = fetched["ref_rows"]
    mem_rows = fetched["mem_rows"]

    # The returned docs from pinecone are already dense-ranked. We can use that ordering.
    ref_texts = _texts(ref_rows)
    mem_texts = _texts(mem_rows)
    
    # Dense results are the rows in the order returned by Pinecone
    ref_dense = ref_rows 
    mem_dense = mem_rows

    # Sparse results from BM25
    bm25_ref = BM25Okapi([t.split() for t in ref_texts]) if len(ref_texts) >= 10 else None
    bm25_mem = BM25Okapi([t.split() for t in mem_texts]) if len(mem_texts) >= 10 else None
    
    ref_sparse_idx = _topk_bm25(bm25_ref, query, min(top_ref, len(ref_texts))) if bm25_ref else []
    ref_sparse = [ref_rows[i] for i in ref_sparse_idx]
    
    mem_sparse_idx = _topk_bm25(bm25_mem, query, min(top_mem, len(mem_texts))) if bm25_mem else []
    mem_sparse = [mem_rows[i] for i in mem_sparse_idx]

    # Combine dense and sparse results
    # This is a simplified RRF, prioritizing dense results.
    combined_refs = _dedupe_rows_keep_order(ref_dense + ref_sparse)
    combined_mems = _dedupe_rows_keep_order(mem_dense + mem_sparse)
    
    refs = combined_refs[:top_ref * 2]
    mems = combined_mems[:top_mem * 2]

    # Pool then optional cross-encoder re-rank
    pool = refs + mems
    
    # BOTTLENECK #3 FIX: Cap pool size before reranking
    if use_cross_encoder and pool:
        pool = pool[:12] # Cap the pool to a reasonable size for the cross-encoder
        ce_order = _cross_encoder_rerank(query, pool)
        pool = [pool[i] for i in ce_order]

    # MMR step removed as it requires vectors, which we are no longer fetching.
    
    selected_refs = [x for x in pool if x.get("type") == "reflection"][:top_ref]
    selected_mems = [x for x in pool if x.get("type") == "memory"][:top_mem]

    TIMINGS['influencer_retrieve'] = time.time() - t0
    return {
        "lenses_used": lenses,
        "reflections": selected_refs,
        "memories": selected_mems,
    }


# TEMPLATE constant removed - now using dynamic prompt from YAML

def _truncate_lines(lines: List[str], max_chars: int) -> List[str]:
    out, total = [], 0
    for ln in lines:
        if total + len(ln) > max_chars:
            break
        out.append(ln)
        total += len(ln)
    return out


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
         "content": "You are a helpful assistant who speaks in the influencer’s authentic voice while staying factual."},
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


def generate_conversation_response(state: State) -> State:
    system_prompt = prompt_templates['MAIN_SYSTEM_PROMPT']
    system_prompt = system_prompt.format(summaries=state['retrieved_summaries'])
    # Send only the latest configured messages to the conversation model for efficiency
    history_to_send = state.get('chat_history', [])[-PAST_CHAT_HISTORY_CNT:]
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *history_to_send
    ])
    return {"conv_response": str(response.content)}


def generate_influencer_answer(state: State) -> State:
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


def summarize(state: State) -> State:
    # Summarize based on total message count (user + assistant) reaching multiples of threshold
    try:
        total_msgs_cnt = len(state.get('chat_history', []))
    except Exception:
        total_msgs_cnt = 0
    # Trigger when either the current total reaches a multiple of threshold, or
    # when adding the imminent assistant reply would reach it. This handles
    # different caller timings between local and prod flows.
    hits_now = (total_msgs_cnt % CONVERSATION_SUMMARY_THRESHOLD == 0)
    hits_with_next = ((total_msgs_cnt + 1) % CONVERSATION_SUMMARY_THRESHOLD == 0)
    is_summary_turn = total_msgs_cnt > 0 and (hits_now or hits_with_next)

    if not is_summary_turn:
        return {"summary_generated": False, "message_summary": ""}

    # Summarize the last N messages (by total messages)
    messages_to_summarize_txt = messages_to_txt(state.get('chat_history', [])[-CONVERSATION_SUMMARY_THRESHOLD:])

    prompt = prompt_templates['SUMMARY_PROMPT']
    prompt = prompt.format(
        conversation=messages_to_summarize_txt
    )

    response = llm.invoke(prompt)

    # Store in vector DB for retrieval
    metadata_payload = {
        "text": response.content,
        "user_id": state['user_id']
    }

    vectors = [(
        str(uuid.uuid4()),
        embeddings.embed_query(response.content),
        metadata_payload
    )]
    index.upsert(vectors=vectors)

    return {"message_summary": str(response.content), "summary_generated": True}


# -------------------------
# Security and Content Moderation Functions
# -------------------------

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


graph_builder = StateGraph(State)

graph_builder.add_node(retrieve_context)
graph_builder.add_node(generate_influencer_answer)
graph_builder.add_node(security_check_node)
graph_builder.add_node(regenerate_safe_response)
graph_builder.add_node(summarize)

graph_builder.add_edge(START, 'retrieve_context')
graph_builder.add_edge('retrieve_context', 'generate_influencer_answer')
graph_builder.add_edge('generate_influencer_answer', 'security_check_node')

# Conditional edge: if security check fails, retry or proceed to summarize
graph_builder.add_conditional_edges(
    'security_check_node',
    should_retry_security,
    {
        "regenerate_safe_response": "regenerate_safe_response",
        "summarize": "summarize"
    }
)

# After regenerating, check security again
graph_builder.add_edge('regenerate_safe_response', 'security_check_node')
graph_builder.add_edge('summarize', END)

chatbot_clio = graph_builder.compile()
