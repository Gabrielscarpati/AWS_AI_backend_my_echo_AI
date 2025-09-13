# retrieval.py
# Query-time functions: lens routing, hybrid retrieval (dense + BM25), optional cross-encoder re-rank,
# MMR for diversity, prompt assembly, and the "G" (generation) for full RAG.

import os, json, argparse
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# --------- Optional (quality): Cross-encoder re-ranker ----------
# If installed, we'll use it; if not, we gracefully skip it.
_CE = None
def _maybe_load_cross_encoder():
    global _CE
    if _CE is not None:
        return _CE
    try:
        from sentence_transformers import CrossEncoder
        # small, fast MS MARCO model; swap if you want larger quality
        _CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        _CE = None
    return _CE

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
INDEX_DIR = Path("indexes")
MEM_FILE = INDEX_DIR / "memories.json"
REF_FILE = INDEX_DIR / "reflections.json"

# -------------------------
# Lens router (rules; replace with classifier if needed)
# -------------------------
LENS_KEYWORDS = {
    "behav_econ": ["price","budget","cpm","roi","usage","rights","whitelist","format","deliverable","contract","payment","sponsorship","accept","deal"],
    "psych":      ["tone","voice","style","boundaries","values","ethics","creative","script","tone of voice","dm","intro"],
    "political":  ["politics","controversial","avoid","red line","endorsement","mlm","diet"],
    "demo":       ["audience","demographic","age","country","region","when","time","timezone","peak","engagement"],
}
ORDER = ["behav_econ","psych","demo","political"]  # fallback priority

def pick_lenses(q: str, top: int = 2) -> List[str]:
    ql = q.lower()
    scores = {k:0 for k in LENS_KEYWORDS}
    for lens, kws in LENS_KEYWORDS.items():
        for kw in kws:
            if kw in ql:
                scores[lens] += 1
    hit = [k for k,v in scores.items() if v>0]
    lenses = hit if hit else ORDER
    return lenses[:top]

# -------------------------
# Load indexes
# -------------------------
def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _texts(rows: List[Dict[str, Any]]) -> List[str]:
    # reflections use 'bullet', memories use 'text'
    out = []
    for r in rows:
        out.append(r["bullet"] if r.get("type") == "reflection" else r["text"])
    return out

def _emb_matrix(rows: List[Dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 384), dtype=np.float32)  # shape won't matter if empty
    arr = np.array([r["embedding"] for r in rows], dtype=np.float32)
    # assume already normalized at indexing time
    return arr

# -------------------------
# Retrieval helpers
# -------------------------
_model = None
def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed_query(q: str) -> np.ndarray:
    return _get_model().encode([q], normalize_embeddings=True)[0]

def topk_dense(qv: np.ndarray, vecs: np.ndarray, k: int) -> List[int]:
    if vecs.shape[0] == 0 or k <= 0:
        return []
    sims = vecs @ qv  # cosine if normalized
    idx = np.argsort(sims)[-k:][::-1]
    return idx.tolist()

def topk_bm25(bm25: BM25Okapi, query: str, k: int) -> List[int]:
    if bm25 is None or k <= 0:
        return []
    toks = query.split()
    scores = bm25.get_scores(toks)
    idx = np.argsort(scores)[-k:][::-1]
    return idx.tolist()

def mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, lambda_mult=0.7, k=10) -> List[int]:
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

def dedupe_keep_order(items: List[int]) -> List[int]:
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out

def cross_encoder_rerank(query: str, pool_rows: List[Dict[str, Any]]) -> List[int]:
    """
    Optional quality boost: re-rank pooled items with a cross-encoder.
    Returns indices (sorted by descending relevance).
    """
    CE = _maybe_load_cross_encoder()
    if CE is None or not pool_rows:
        return list(range(len(pool_rows)))
    texts = []
    for r in pool_rows:
        txt = r["bullet"] if r.get("type") == "reflection" else r["text"]
        texts.append(txt)
    pairs = [[query, t] for t in texts]
    scores = CE.predict(pairs)  # higher is better
    order = np.argsort(scores)[::-1].tolist()
    return order

# -------------------------
# Main retrieval pipeline
# -------------------------
def retrieve(
    query: str,
    creator_id: str,
    top_ref: int = 5,
    top_mem: int = 8,
    use_cross_encoder: bool = True,
) -> Dict[str, Any]:
    # Load indexes
    all_refs = [r for r in _load_rows(REF_FILE) if r.get("creator_id") == creator_id]
    all_mems = [m for m in _load_rows(MEM_FILE) if m.get("creator_id") == creator_id]

    # Lens routing for reflections
    lenses = pick_lenses(query)
    ref_pool = [r for r in all_refs if r.get("role") in lenses] or all_refs

    # Build matrices & BM25
    ref_texts = _texts(ref_pool)
    mem_texts = _texts(all_mems)
    ref_vecs = _emb_matrix(ref_pool)
    mem_vecs = _emb_matrix(all_mems)
    bm25_ref = BM25Okapi([t.split() for t in ref_texts]) if ref_texts else None
    bm25_mem = BM25Okapi([t.split() for t in mem_texts]) if mem_texts else None

    qv = embed_query(query)

    # Hybrid: reflections
    ref_dense_idx = topk_dense(qv, ref_vecs, min(top_ref, ref_vecs.shape[0]))
    ref_sparse_idx = topk_bm25(bm25_ref, query, min(top_ref, len(ref_texts))) if bm25_ref else []
    ref_idxs = dedupe_keep_order(ref_dense_idx + ref_sparse_idx)[:top_ref]
    refs = [ref_pool[i] for i in ref_idxs]

    # Hybrid: memories
    mem_dense_idx = topk_dense(qv, mem_vecs, min(top_mem, mem_vecs.shape[0]))
    mem_sparse_idx = topk_bm25(bm25_mem, query, min(top_mem, len(mem_texts))) if bm25_mem else []
    mem_idxs = dedupe_keep_order(mem_dense_idx + mem_sparse_idx)[:top_mem]
    mems = [all_mems[i] for i in mem_idxs]

    # Pool → optional cross-encoder re-rank → then MMR diversity
    pool = refs + mems
    if use_cross_encoder and pool:
        ce_order = cross_encoder_rerank(query, pool)
        pool = [pool[i] for i in ce_order]

    if pool:
        pool_vecs = np.vstack([
            *(np.array([ (r["embedding"]) for r in pool ], dtype=np.float32)),
        ])
        sel = mmr(qv, pool_vecs, lambda_mult=0.7, k=min(10, pool_vecs.shape[0]))
        pooled = [pool[i] for i in sel]
    else:
        pooled = []

    # Split back
    selected_refs = [x for x in pooled if x.get("type") == "reflection"][:top_ref]
    selected_mems = [x for x in pooled if x.get("type") == "memory"][:top_mem]

    return {
        "lenses_used": lenses,
        "reflections": selected_refs,
        "memories": selected_mems,
    }

# -------------------------
# Prompt assembly (+ token budget guard)
# -------------------------
TEMPLATE = """You are preparing an answer about influencer {creator_id}.
Use REFLECTIONS for stance/values/best practices; use MEMORIES for quotes/examples/dates.
Cite short source notes (source/date) in-line. Annotate claims with [r:ID] or [m:ID] where helpful.

User question:
{question}

Lenses selected: {lenses}

REFLECTIONS (role • theme • bullet • id):
{ref_lines}

MEMORIES (source @ date • id) snippet:
{mem_lines}

Answer with: (a) 3–5 bullet recommendations (b) 1–2 tailored examples (c) any red lines to avoid.
"""

def _truncate_lines(lines: List[str], max_chars: int) -> List[str]:
    out, total = [], 0
    for ln in lines:
        if total + len(ln) > max_chars:
            break
        out.append(ln)
        total += len(ln)
    return out

def format_pack(creator_id: str, question: str, pack: Dict[str, Any], max_chars: int = 6000) -> str:
    # include IDs so the model can cite them (helps traceability)
    ref_lines_all = [f"- [{r['role']}] • {r['theme']} • {r['bullet']} • id={r['id']}" for r in pack["reflections"]]
    mem_lines_all = [f"- ({m['source']} @ {m['created_at']}) {m['text']} • id={m['id']}" for m in pack["memories"]]
    # simple character budget
    ref_lines = _truncate_lines(ref_lines_all, max_chars // 2)
    mem_lines = _truncate_lines(mem_lines_all, max_chars // 2)
    return TEMPLATE.format(
        creator_id=creator_id,
        question=question,
        lenses=", ".join(pack["lenses_used"]),
        ref_lines="\n".join(ref_lines) or "- (none)",
        mem_lines="\n".join(mem_lines) or "- (none)",
    )

# -------------------------
# RAG: call an LLM (OpenAI or local Ollama with OpenAI-compatible endpoint)
# -------------------------
def _openai_chat(messages: List[Dict[str,str]], model: str, temperature: float, max_tokens: int) -> str:
    # Uses the official OpenAI client (pip install openai>=1.0.0)
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
    # Works with Ollama's OpenAI-compatible /v1/chat/completions (ollama serve; OLLAMA_BASE_URL env)
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
    model: str = None,
    provider: str = None,
    temperature: float = 0.4,
    max_tokens: int = 600,
    use_cross_encoder: bool = True,
) -> Dict[str, Any]:
    """
    Full RAG: retrieve -> assemble -> generate.
    provider: "openai" or "ollama" (auto-detect if None: prefer openai if key set).
    """
    # 1) Retrieve
    pack = retrieve(
        question, creator_id=creator_id,
        use_cross_encoder=use_cross_encoder
    )
    prompt = format_pack(creator_id, question, pack)

    # 2) Choose provider/model
    provider = provider or ("openai" if os.getenv("OPENAI_API_KEY") else "ollama")
    if provider == "openai":
        model = model or os.getenv("OPENAI_RAG_MODEL", "gpt-4.1-mini")
    else:
        model = model or os.getenv("OLLAMA_RAG_MODEL", "llama3.1")

    # 3) Messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant who speaks in the influencer’s authentic voice while staying factual."},
        {"role": "user", "content": prompt}
    ]

    # 4) Generate
    if provider == "openai":
        text = _openai_chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        text = _ollama_chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)

    return {
        "provider": provider,
        "model": model,
        "question": question,
        "lenses_used": pack["lenses_used"],
        "reflections": pack["reflections"],
        "memories": pack["memories"],
        "answer": text
    }

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", type=str, required=False, help="User question")
    parser.add_argument("--creator", type=str, default="ava_rivera")
    parser.add_argument("--rag", action="store_true", help="Generate final answer with RAG")
    parser.add_argument("--provider", type=str, default=None, help="openai | ollama (auto if omitted)")
    parser.add_argument("--model", type=str, default=None, help="override model name")
    parser.add_argument("--no_ce", action="store_true", help="disable cross-encoder re-rank")
    parser.add_argument("--max_tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.4)
    args = parser.parse_args()

    questions = [args.ask] if args.ask else [
        "What sponsorships would she likely accept and how should we structure the deal?",
        "How should we write the first DM—tone and key points?",
        "Any topics to avoid when pitching her?",
        "When is the best posting window for the US audience?",
    ]

    for q in questions:
        if args.rag:
            out = answer_with_rag(
                q, creator_id=args.creator,
                model=args.model, provider=args.provider,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                use_cross_encoder=(not args.no_ce)
            )
            print("="*88)
            print(f"Q: {q}\n")
            print(out["answer"], "\n")
            # Compact source list (IDs) for traceability
            r_ids = ", ".join([r["id"] for r in out["reflections"]])
            m_ids = ", ".join([m["id"] for m in out["memories"]])
            print(f"Sources → reflections: [{r_ids}]  memories: [{m_ids}]  (lenses: {', '.join(out['lenses_used'])})\n")
        else:
            pack = retrieve(q, creator_id=args.creator, use_cross_encoder=(not args.no_ce))
            prompt = format_pack(args.creator, q, pack)
            print("="*88)
            print(prompt)
            print()
