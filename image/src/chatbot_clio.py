from dotenv import load_dotenv

load_dotenv()

import yaml
import uuid
import re
import os


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


with open("prompt_templates.yaml") as prompt_template_file:
    prompt_templates = yaml.safe_load(prompt_template_file)

flags = re.DOTALL | re.MULTILINE
PATTERN_USER = re.compile(r"<user>(.*?)<\/user>", flags=flags)
PATTERN_ASSISTANT = re.compile(r"<assistant>(.*?)<\/assistant>", flags=flags)

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

RETRIEVAL_SUMMARY_CNT = 3
CONVERSATION_SUMMARY_THRESHOLD = 10
PAST_CHAT_HISTORY_CNT = 19
EMBEDDING_DIMENSION = 1024

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
    chat_history: List[BaseMessage]
    msgs_cnt_by_user: int
    user_query: str
    retrieved_summaries: str
    conv_response: str
    influencer_answer: str
    influencer_sources: Dict[str, Any]
    response: str
    is_summary_turn: bool
    message_summary: str
    summary_generated: bool


def messages_to_txt(messages: List[BaseMessage]) -> str:
    role_map = {
        'human': 'user',
        'ai': 'assistant',
    }
    return "\n".join([
        f"{role_map[message.type].upper()}: {message.content}"
        for message in messages
    ])


def construct_user_query(state: State) -> State:
    prompt = prompt_templates['USER_QUERY_CONSTRUCTION_PROMPT']

    chat_history = state['chat_history']

    prompt = prompt.format(
        conversation=messages_to_txt(chat_history[:-1]),
        user_input=chat_history[-1].content
    )

    response = llm.invoke(prompt)

    return {'user_query': str(response.content)}


def retrieve_context(state: State) -> State:
    """Retrieves relevant summaries of conversation from the vector store"""
    retrieved_docs = vector_store.similarity_search(
        state["user_query"], 
        k=RETRIEVAL_SUMMARY_CNT,
        filter={"user_id": state["user_id"]}
    )
    
    retrieved_summaries = "\n\n".join([
        f"SUMMARY {i}:\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs, start=1)
    ])
    return {"retrieved_summaries": retrieved_summaries}


# -------------------------
# Influencer RAG helpers (retrieve + prompt assembly + generation)
# -------------------------

_model_st: SentenceTransformer | None = None
def _get_st_model():
    global _model_st
    if _model_st is None:
        _model_st = SentenceTransformer(EMBEDDING_MODEL)
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
    "behav_econ": ["price","budget","cpm","roi","usage","rights","whitelist","format","deliverable","contract","payment","sponsorship","accept","deal"],
    "psych":      ["tone","voice","style","boundaries","values","ethics","creative","script","tone of voice","dm","intro"],
    "political":  ["politics","controversial","avoid","red line","endorsement","mlm","diet"],
    "demo":       ["audience","demographic","age","country","region","when","time","timezone","peak","engagement"],
}
ORDER = ["behav_econ","psych","demo","political"]

def _pick_lenses(q: str, top: int = 2) -> List[str]:
    ql = q.lower()
    scores = {k:0 for k in LENS_KEYWORDS}
    for lens, kws in LENS_KEYWORDS.items():
        for kw in kws:
            if kw in ql:
                scores[lens] += 1
    hit = [k for k,v in scores.items() if v>0]
    lenses = hit if hit else ORDER
    return lenses[:top]

def influencer_retrieve(query: str, creator_id: str, top_ref: int = 5, top_mem: int = 8, use_cross_encoder: bool = True) -> Dict[str, Any]:
    lenses = _pick_lenses(query)
    qv = _embed_query(query)

    def _from_pinecone() -> Dict[str, Any] | None:
        if influencer_index is None:
            return None
        try:
            # Fetch a wider candidate pool by type; then hybrid re-rank locally
            ref_filter = {"creator_id": creator_id, "type": "reflection"}
            if lenses:
                ref_filter["role"] = {"$in": lenses}
            mem_filter = {"creator_id": creator_id, "type": "memory"}

            ref_res = influencer_index.query(vector=qv.tolist(), top_k=max(top_ref*5, 10), filter=ref_filter, include_metadata=True)
            mem_res = influencer_index.query(vector=qv.tolist(), top_k=max(top_mem*5, 16), filter=mem_filter, include_metadata=True)

            def _rows_from_matches(res, kind: str) -> List[Dict[str, Any]]:
                matches = getattr(res, "matches", None) or res.get("matches", [])
                rows: List[Dict[str, Any]] = []
                for m in matches:
                    md = getattr(m, "metadata", None) or m.get("metadata", {})
                    rid = getattr(m, "id", None) or m.get("id")
                    if kind == "reflection":
                        rows.append({
                            "type": "reflection",
                            "id": rid,
                            "creator_id": md.get("creator_id"),
                            "role": md.get("role"),
                            "theme": md.get("theme"),
                            "bullet": md.get("bullet", ""),
                            "source_ids": md.get("source_ids", []),
                            "created_at": md.get("created_at"),
                        })
                    else:
                        rows.append({
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
                        })
                return rows

            ref_rows = _rows_from_matches(ref_res, "reflection")
            mem_rows = _rows_from_matches(mem_res, "memory")
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

    # Build hybrid signals locally (dense + BM25) on fetched candidates
    ref_texts = _texts(ref_rows)
    mem_texts = _texts(mem_rows)
    ref_vecs = _get_st_model().encode(ref_texts, normalize_embeddings=True) if ref_texts else np.zeros((0,384), dtype=np.float32)
    mem_vecs = _get_st_model().encode(mem_texts, normalize_embeddings=True) if mem_texts else np.zeros((0,384), dtype=np.float32)
    bm25_ref = BM25Okapi([t.split() for t in ref_texts]) if ref_texts else None
    bm25_mem = BM25Okapi([t.split() for t in mem_texts]) if mem_texts else None

    ref_dense_idx = _topk_dense(qv, np.array(ref_vecs, dtype=np.float32), min(top_ref, len(ref_texts)))
    ref_sparse_idx = _topk_bm25(bm25_ref, query, min(top_ref, len(ref_texts))) if bm25_ref else []
    ref_idxs = _dedupe_keep_order(ref_dense_idx + ref_sparse_idx)[:top_ref*2]
    refs = [ref_rows[i] for i in ref_idxs] if ref_rows else []

    mem_dense_idx = _topk_dense(qv, np.array(mem_vecs, dtype=np.float32), min(top_mem, len(mem_texts)))
    mem_sparse_idx = _topk_bm25(bm25_mem, query, min(top_mem, len(mem_texts))) if bm25_mem else []
    mem_idxs = _dedupe_keep_order(mem_dense_idx + mem_sparse_idx)[:top_mem*2]
    mems = [mem_rows[i] for i in mem_idxs] if mem_rows else []

    # Pool then optional cross-encoder re-rank
    pool = refs + mems
    if use_cross_encoder and pool:
        ce_order = _cross_encoder_rerank(query, pool)
        pool = [pool[i] for i in ce_order]

    # MMR diversity using locally computed vectors
    if pool:
        def _key(r: Dict[str, Any]) -> str:
            return f"{r.get('type','?')}::{r.get('id','')}"
        vec_map: Dict[str, np.ndarray] = {}
        for i, r in enumerate(ref_rows):
            if i < len(ref_vecs):
                vec_map[_key(r)] = np.array(ref_vecs[i], dtype=np.float32)
        for i, r in enumerate(mem_rows):
            if i < len(mem_vecs):
                vec_map[_key(r)] = np.array(mem_vecs[i], dtype=np.float32)
        pool_vecs = np.vstack([vec_map.get(_key(r), np.zeros(384, dtype=np.float32)) for r in pool]) if pool else np.zeros((0,384), dtype=np.float32)
        sel = _mmr(qv, pool_vecs, lambda_mult=0.7, k=min(10, pool_vecs.shape[0]))
        pooled = [pool[i] for i in sel]
    else:
        pooled = []

    selected_refs = [x for x in pooled if x.get("type") == "reflection"][:top_ref]
    selected_mems = [x for x in pooled if x.get("type") == "memory"][:top_mem]

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

def format_pack(creator_id: str, question: str, pack: Dict[str, Any], influencer_name: str = None, conversation_summaries: str = "", max_chars: int = 6000) -> str:
    ref_lines_all = [f"- [{r['role']}] • {r.get('theme','')} • {r.get('bullet','')} • id={r['id']}" for r in pack.get("reflections", [])]
    mem_lines_all = [f"- ({m.get('source','')} @ {m.get('created_at','')}) {m.get('text','')} • id={m['id']}" for m in pack.get("memories", [])]
    ref_lines = _truncate_lines(ref_lines_all, max_chars // 2)
    mem_lines = _truncate_lines(mem_lines_all, max_chars // 2)
    
    # Use influencer name if provided, otherwise fallback to creator_id
    display_name = influencer_name or creator_id
    
    # Use the dynamic template from YAML
    template = prompt_templates['DYNAMIC_INFLUENCER_PROMPT']
    return template.format(
        influencer_name=display_name,
        question=question,
        ref_lines="\n".join(ref_lines) or "- (none)",
        mem_lines="\n".join(mem_lines) or "- (none)",
        conversation_summaries=conversation_summaries,
    )

def _openai_chat(messages: List[Dict[str,str]], model: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI()
    res = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return res.choices[0].message.content

def _ollama_chat(messages: List[Dict[str,str]], model: str, temperature: float, max_tokens: int) -> str:
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

def answer_with_rag(question: str, creator_id: str, influencer_name: str = None, conversation_summaries: str = "", model: str | None = None, provider: str | None = None, temperature: float = 0.4, max_tokens: int = 600, use_cross_encoder: bool = True) -> Dict[str, Any]:
    pack = influencer_retrieve(question, creator_id=creator_id, use_cross_encoder=use_cross_encoder)
    prompt = format_pack(creator_id, question, pack, influencer_name=influencer_name, conversation_summaries=conversation_summaries)

    provider = provider or ("openai" if os.getenv("OPENAI_API_KEY") else "ollama")
    if provider == "openai":
        model = model or os.getenv("OPENAI_RAG_MODEL", "gpt-4.1-mini")
    else:
        model = model or os.getenv("OLLAMA_RAG_MODEL", "llama3.1")

    messages = [
        {"role": "system", "content": "You are a helpful assistant who speaks in the influencer’s authentic voice while staying factual."},
        {"role": "user", "content": prompt}
    ]

    if provider == "openai":
        text = _openai_chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        text = _ollama_chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)

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
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state['chat_history']
    ])
    return {"conv_response": str(response.content)}


def generate_influencer_answer(state: State) -> State:
    creator_id = state.get('creator_id') or ""
    # Get influencer name from state or use creator_id as fallback
    influencer_name = state.get('influencer_name') or creator_id
    
    # Pass conversation summaries separately rather than combining with question
    user_question = state.get('user_query', '')
    conversation_summaries = state.get('retrieved_summaries', '')
    
    out = answer_with_rag(
        user_question,
        creator_id=creator_id,
        influencer_name=influencer_name,
        conversation_summaries=conversation_summaries,
        temperature=float(os.getenv("INFLUENCER_RAG_TEMPERATURE", 0.4)),
        max_tokens=int(os.getenv("INFLUENCER_RAG_MAX_TOKENS", 600)),
        use_cross_encoder=os.getenv("INFLUENCER_RAG_USE_CE", "true").lower() != "false",
    )
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
    }


def summarize(state: State) -> State:
    # Prefer explicit frontend flag; otherwise fallback to modulo logic
    is_summary_turn = state.get('is_summary_turn')
    if is_summary_turn is None:
        msgs_cnt_by_user = state.get('msgs_cnt_by_user', 0)
        is_summary_turn = msgs_cnt_by_user > 0 and (msgs_cnt_by_user % CONVERSATION_SUMMARY_THRESHOLD == 0)

    if not is_summary_turn:
        return {"summary_generated": False, "message_summary": ""}

    # Summarize the last N messages
    messages_to_summarize_txt = messages_to_txt(state['chat_history'][-CONVERSATION_SUMMARY_THRESHOLD:])

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


graph_builder = StateGraph(State)

graph_builder.add_node(construct_user_query)
graph_builder.add_node(retrieve_context)
graph_builder.add_node(generate_influencer_answer)
graph_builder.add_node(summarize)

graph_builder.add_edge(START, 'construct_user_query')
graph_builder.add_edge('construct_user_query', 'retrieve_context')
graph_builder.add_edge('retrieve_context', 'generate_influencer_answer')
graph_builder.add_edge('generate_influencer_answer', 'summarize')
graph_builder.add_edge('summarize', END)

chatbot_clio = graph_builder.compile()