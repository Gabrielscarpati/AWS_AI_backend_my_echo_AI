import json
import time
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from .config import EMBEDDING_MODEL
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
        # Prefer full_text if present (e.g., interview analysis), otherwise fallback to text
        txt = r.get("full_text") or r.get("text", "")
        out.append(txt)
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
        txt = r.get("full_text") or r.get("text", "")
        texts.append(txt)
    pairs = [[query, t] for t in texts]
    scores = CE.predict(pairs)
    order = np.argsort(scores)[::-1].tolist()
    return order


def _pick_lenses(q: str, top: int = 2) -> List[str]:
    # Lenses are no longer used with the new categories, keep placeholder for compatibility
    return []


def influencer_retrieve(query: str, creator_id: str,
                        fetch_per_category: int = 6, final_per_category: int = 3,
                        use_cross_encoder: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    qv = _embed_query(query)

    def _from_pinecone() -> Dict[str, Any] | None:
        if influencer_index is None:
            return None
        try:
            # Query per category in parallel
            def _query_category(category: str, top_k: int) -> List[Dict[str, Any]]:
                res = influencer_index.query(
                    vector=qv.tolist(),
                    top_k=top_k,
                    filter={"creator_id": creator_id, "category": category},
                    include_metadata=True,
                    include_values=False,
                )
                matches = getattr(res, "matches", None) or res.get("matches", [])
                rows: List[Dict[str, Any]] = []
                for m in matches:
                    md = getattr(m, "metadata", None) or m.get("metadata", {})
                    rid = getattr(m, "id", None) or m.get("id")
                    row: Dict[str, Any] = {
                        "id": rid,
                        "creator_id": md.get("creator_id"),
                        "category": md.get("category"),
                        "privacy_level": md.get("privacy_level"),
                        "timestamp": md.get("timestamp"),
                    }
                    if category == "context_data":
                        row.update({
                            "type": "context_data",
                            "text": md.get("text", ""),
                            "source": md.get("source", ""),
                        })
                    elif category == "expert_analysis":
                        row.update({
                            "type": "expert_analysis",
                            "text": md.get("text", ""),
                            "title": md.get("title", ""),
                        })
                    elif category == "interview_and_communication_style":
                        row.update({
                            "type": "interview_and_communication_style",
                            "text": md.get("text", ""),
                            "full_text": md.get("full_text", ""),
                            "model_type": md.get("model_type", ""),
                            "sub_category": md.get("sub_category", ""),
                            "scores": md.get("scores", ""),
                            "title": md.get("title", ""),
                        })
                    rows.append(row)
                return rows

            with ThreadPoolExecutor(max_workers=3) as ex:
                futs = {
                    ex.submit(_query_category, "context_data", fetch_per_category): "context_data",
                    ex.submit(_query_category, "expert_analysis", fetch_per_category): "expert_analysis",
                    ex.submit(_query_category, "interview_and_communication_style", fetch_per_category): "interview_and_communication_style",
                }
                out: Dict[str, List[Dict[str, Any]]] = {
                    "context_rows": [],
                    "expert_rows": [],
                    "interview_rows": [],
                }
                for fut in as_completed(futs):
                    cat = futs[fut]
                    rows = fut.result()
                    if cat == "context_data":
                        out["context_rows"] = rows
                    elif cat == "expert_analysis":
                        out["expert_rows"] = rows
                    else:
                        out["interview_rows"] = rows

            return out
        except Exception:
            return None

    def _from_json() -> Dict[str, Any]:
        # No JSON fallback for new categories; return empty
        return {"context_rows": [], "expert_rows": [], "interview_rows": []}

    fetched = _from_pinecone() or _from_json()
    context_rows = fetched["context_rows"]
    expert_rows = fetched["expert_rows"]
    interview_rows = fetched["interview_rows"]

    # The returned docs from pinecone are already dense-ranked. We can use that ordering.
    context_texts = _texts(context_rows)
    expert_texts = _texts(expert_rows)
    interview_texts = _texts(interview_rows)
    
    # Dense results are the rows in the order returned by Pinecone
    context_dense = context_rows
    expert_dense = expert_rows
    interview_dense = interview_rows

    # Sparse results from BM25 (run in parallel)
    def _bm25_sparse(rows: List[Dict[str, Any]], texts: List[str], top_k: int) -> List[Dict[str, Any]]:
        bm25 = BM25Okapi([t.split() for t in texts]) if len(texts) >= 10 else None
        idxs = _topk_bm25(bm25, query, min(top_k, len(texts))) if bm25 else []
        return [rows[i] for i in idxs]

    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_c = ex.submit(_bm25_sparse, context_dense, context_texts, final_per_category)
        fut_e = ex.submit(_bm25_sparse, expert_dense, expert_texts, final_per_category)
        fut_i = ex.submit(_bm25_sparse, interview_dense, interview_texts, final_per_category)
        context_sparse = fut_c.result()
        expert_sparse = fut_e.result()
        interview_sparse = fut_i.result()

    # Combine dense and sparse results
    # This is a simplified RRF, prioritizing dense results.
    combined_context = _dedupe_rows_keep_order(context_dense + context_sparse)
    combined_expert = _dedupe_rows_keep_order(expert_dense + expert_sparse)
    combined_interview = _dedupe_rows_keep_order(interview_dense + interview_sparse)

    contexts = combined_context[: min(len(combined_context), fetch_per_category)]
    experts = combined_expert[: min(len(combined_expert), fetch_per_category)]
    interviews = combined_interview[: min(len(combined_interview), fetch_per_category)]

    # Optional cross-encoder re-rank per category, run in parallel
    def _rerank_and_take(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []
        if use_cross_encoder:
            cap = rows[: min(len(rows), 12)]
            order = _cross_encoder_rerank(query, cap)
            rows = [cap[i] for i in order]
        return rows[: final_per_category]

    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_rc = ex.submit(_rerank_and_take, contexts)
        fut_re = ex.submit(_rerank_and_take, experts)
        fut_ri = ex.submit(_rerank_and_take, interviews)
        selected_contexts = fut_rc.result()
        selected_experts = fut_re.result()
        selected_interviews = fut_ri.result()

    # MMR step removed as it requires vectors, which we are no longer fetching.

    TIMINGS['influencer_retrieve'] = time.time() - t0
    return {
        "lenses_used": [],
        "context_data": selected_contexts,
        "expert_analysis": selected_experts,
        "interview_styles": selected_interviews,
    }
