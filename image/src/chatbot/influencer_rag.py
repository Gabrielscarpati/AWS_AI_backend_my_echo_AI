import json
import time
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from .config import EMBEDDING_MODEL, INDEX_DIR, MEM_FILE, REF_FILE, LENS_KEYWORDS, ORDER
from .retrieval import influencer_index, TIMINGS

_CE = None
_model_st: SentenceTransformer = SentenceTransformer(EMBEDDING_MODEL)


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
