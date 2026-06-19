"""Retriever candidate-cache builders for reranking."""

import logging
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
from sentence_transformers import SentenceTransformer

from ..core.dense import get_dense_topk_batched
from lib.utils.logging_utils import log_timed


logger = logging.getLogger(__name__)


def _build_query_primary_key(document_id, spans) -> str:
    return f"{document_id}|{spans}"


def _ensure_candidate_text_map(candidate_text_map: Dict[tuple[str, str], str] | None) -> Dict[tuple[str, str], str]:
    return {} if candidate_text_map is None else candidate_text_map


def _normalize_text_key(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


def _format_candidate_text_with_matched_alias(
    *,
    base_candidate_text: str,
    matched_alias_text: str,
) -> str:
    base_candidate_text = str(base_candidate_text or "").strip()
    matched_alias_text = str(matched_alias_text or "").strip()

    if not matched_alias_text:
        return base_candidate_text
    if not base_candidate_text:
        return matched_alias_text
    if _normalize_text_key(base_candidate_text) == _normalize_text_key(matched_alias_text):
        return base_candidate_text
    base_segments = [_normalize_text_key(segment) for segment in base_candidate_text.split(";")]
    if _normalize_text_key(matched_alias_text) in base_segments:
        return base_candidate_text
    return f"{matched_alias_text}; {base_candidate_text}".strip()


def build_retriever_candidate_cache(
    entities_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    retriever_model: SentenceTransformer,
    *,
    mention_column: str = "text",
    topk: int,
    query_batch_size: int,
    dense_vocab_batch_size: int,
    st_encode_batch_size: int,
    deduplicate_by_cui: bool = True,
) -> Dict[str, Dict]:
    """Build dense retriever candidate caches for mentions."""
    required_columns = {mention_column, "document_id", "spans", "entity_type"}
    missing_columns = required_columns.difference(entities_df.columns)
    if missing_columns:
        raise ValueError(f"entities_df is missing required columns: {sorted(missing_columns)}")

    cache: Dict[str, Dict] = {}
    entity_types = sorted(entities_df["entity_type"].dropna().astype(str).unique().tolist())
    logger.info(
        "Preparing retriever candidate cache: num_queries=%d, num_entity_types=%d, topk=%d, mention_column=%s",
        len(entities_df),
        len(entity_types),
        topk,
        mention_column,
    )

    with log_timed(logger, "Retriever candidate cache"):
        for entity_type in entity_types:
            subset_df = entities_df[entities_df["entity_type"].astype(str) == entity_type].reset_index(drop=True)
            subset_vocab = vocab_df[vocab_df["semantic_type"].astype(str) == entity_type].reset_index(drop=True)
            if subset_vocab.empty:
                logger.warning("Skipping entity_type=%s because vocab subset is empty", entity_type)
                continue

            vocab_names = subset_vocab["concept_name"].astype(str).values
            vocab_cuis = subset_vocab["CUI"].astype(str).values
            dense_scores, dense_indices = get_dense_topk_batched(
                query_names=subset_df[mention_column].astype(str).values,
                vocab_names=vocab_names,
                vocab_cuis=vocab_cuis,
                st_model=retriever_model,
                base_k=min(int(topk), len(vocab_names)),
                query_batch_size=query_batch_size,
                vocab_batch_size=dense_vocab_batch_size,
                st_encode_batch_size=st_encode_batch_size,
                deduplicate_by_cui=deduplicate_by_cui,
                show_progress=True,
            )

            cache[entity_type] = {
                "mention_column": str(mention_column),
                "query_texts": subset_df[mention_column].astype(str).tolist(),
                "document_ids": subset_df["document_id"].astype(str).tolist(),
                "spans": subset_df["spans"].astype(str).tolist(),
                "entity_type": entity_type,
                "gold_cuis": subset_df["UMLS_CUI"].astype(str).tolist() if "UMLS_CUI" in subset_df.columns else None,
                "vocab_names": vocab_names,
                "vocab_cuis": vocab_cuis,
                "candidate_scores": dense_scores,
                "candidate_indices": dense_indices,
            }
    return cache


def flatten_retriever_candidate_cache(cache: Dict[str, Dict]) -> pd.DataFrame:
    """Convert a retriever candidate cache into preview rows."""
    rows = []
    for entity_type, payload in cache.items():
        gold_cuis = payload.get("gold_cuis") or [None] * len(payload["document_ids"])
        for query_idx, (document_id, spans, query_text, gold_cui, score_row, index_row) in enumerate(
            zip(
                payload["document_ids"],
                payload["spans"],
                payload["query_texts"],
                gold_cuis,
                payload["candidate_scores"],
                payload["candidate_indices"],
            )
        ):
            for rank, (score, candidate_idx) in enumerate(zip(score_row.tolist(), index_row.tolist()), start=1):
                candidate_idx = int(candidate_idx)
                if candidate_idx < 0:
                    continue
                rows.append(
                    {
                        "query_idx": int(query_idx),
                        "query_pk": _build_query_primary_key(document_id, spans),
                        "document_id": str(document_id),
                        "spans": str(spans),
                        "entity_type": str(entity_type),
                        "query_text": str(query_text),
                        "gold_cui": None if gold_cui is None else str(gold_cui),
                        "candidate_rank": int(rank),
                        "candidate_idx": candidate_idx,
                        "candidate_cui": str(payload["vocab_cuis"][candidate_idx]),
                        "candidate_name": str(payload["vocab_names"][candidate_idx]),
                        "retriever_score": float(score),
                    }
                )
    return pd.DataFrame(rows)


def save_retriever_candidate_cache(cache: Dict[str, Dict], output_dir, *, stem: str) -> Dict[str, str]:
    """Save a retriever candidate cache and preview file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pickle_path = output_dir / f"{stem}.pkl"
    preview_path = output_dir / f"{stem}.tsv"

    with pickle_path.open("wb") as f:
        pickle.dump(cache, f)
    flatten_retriever_candidate_cache(cache).to_csv(preview_path, sep="\t", index=False)

    return {
        "pickle_path": str(pickle_path),
        "preview_path": str(preview_path),
    }


def load_retriever_candidate_cache(cache_path) -> Dict[str, Dict]:
    """Load a pickled retriever candidate cache."""
    with Path(cache_path).open("rb") as f:
        return pickle.load(f)


def _build_gold_text_lookup(
    vocab_df: pd.DataFrame,
    candidate_text_map: Dict[tuple[str, str], str],
) -> Dict[tuple[str, str], list[str]]:
    lookup: Dict[tuple[str, str], list[str]] = {}
    for _, row in vocab_df[["CUI", "semantic_type", "concept_name"]].dropna().iterrows():
        key = (str(row["CUI"]), str(row["semantic_type"]))
        lookup.setdefault(key, []).append(str(row["concept_name"]))
    for key, candidate_text in candidate_text_map.items():
        lookup.setdefault(key, [])
        if candidate_text not in lookup[key]:
            lookup[key].append(candidate_text)
    return lookup


def _resolve_candidate_text(
    *,
    candidate_cui: str,
    entity_type: str,
    fallback_name: str,
    candidate_text_map: Dict[tuple[str, str], str],
) -> str:
    base_candidate_text = str(candidate_text_map.get((candidate_cui, entity_type), fallback_name))
    if (candidate_cui, entity_type) not in candidate_text_map:
        return base_candidate_text
    return _format_candidate_text_with_matched_alias(
        base_candidate_text=base_candidate_text,
        matched_alias_text=fallback_name,
    )
