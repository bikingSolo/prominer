"""Build and persist dense retriever caches for dictionary pretraining queries."""

import logging
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
from sentence_transformers import SentenceTransformer

from lib.utils.logging_utils import log_timed

from ...core.dense import get_dense_topk_batched


logger = logging.getLogger(__name__)


def build_dictionary_pretrain_retriever_cache(
    queries_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    retriever_model: SentenceTransformer,
    *,
    topk: int,
    query_batch_size: int,
    dense_vocab_batch_size: int,
    st_encode_batch_size: int,
    deduplicate_by_cui: bool = True,
) -> Dict[str, Dict]:
    """Retrieve top-k dictionary candidates for each pseudo-query."""
    required_query_columns = {"query_id", "query_text", "CUI", "semantic_type", "split"}
    missing_query_columns = required_query_columns.difference(queries_df.columns)
    if missing_query_columns:
        raise ValueError(f"queries_df is missing required columns: {sorted(missing_query_columns)}")

    required_vocab_columns = {"concept_name", "CUI", "semantic_type"}
    missing_vocab_columns = required_vocab_columns.difference(vocab_df.columns)
    if missing_vocab_columns:
        raise ValueError(f"vocab_df is missing required columns: {sorted(missing_vocab_columns)}")

    cache: Dict[str, Dict] = {}
    entity_types = sorted(queries_df["semantic_type"].dropna().astype(str).unique().tolist())
    logger.info(
        "Preparing dictionary pretrain retriever cache: num_queries=%d, num_entity_types=%d, topk=%d",
        len(queries_df),
        len(entity_types),
        topk,
    )

    with log_timed(logger, "Dictionary pretrain retriever cache"):
        for entity_type in entity_types:
            subset_queries = queries_df[queries_df["semantic_type"].astype(str) == entity_type].reset_index(drop=True)
            subset_vocab = vocab_df[vocab_df["semantic_type"].astype(str) == entity_type].reset_index(drop=True)
            if subset_queries.empty or subset_vocab.empty:
                continue

            vocab_names = subset_vocab["concept_name"].astype(str).values
            vocab_cuis = subset_vocab["CUI"].astype(str).values
            dense_scores, dense_indices = get_dense_topk_batched(
                query_names=subset_queries["query_text"].astype(str).values,
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
                "query_ids": subset_queries["query_id"].astype(str).tolist(),
                "query_texts": subset_queries["query_text"].astype(str).tolist(),
                "gold_cuis": subset_queries["CUI"].astype(str).tolist(),
                "splits": subset_queries["split"].astype(str).tolist(),
                "entity_type": str(entity_type),
                "vocab_names": vocab_names,
                "vocab_cuis": vocab_cuis,
                "candidate_scores": dense_scores,
                "candidate_indices": dense_indices,
            }
    return cache


def flatten_dictionary_pretrain_retriever_cache(cache: Dict[str, Dict]) -> pd.DataFrame:
    """Convert a retriever cache into a tabular preview."""
    rows = []
    for entity_type, payload in cache.items():
        for query_idx, (query_id, query_text, gold_cui, split_name, score_row, index_row) in enumerate(
            zip(
                payload["query_ids"],
                payload["query_texts"],
                payload["gold_cuis"],
                payload["splits"],
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
                        "query_id": str(query_id),
                        "query_text": str(query_text),
                        "gold_cui": str(gold_cui),
                        "semantic_type": str(entity_type),
                        "split": str(split_name),
                        "candidate_rank": int(rank),
                        "candidate_idx": candidate_idx,
                        "candidate_cui": str(payload["vocab_cuis"][candidate_idx]),
                        "candidate_name": str(payload["vocab_names"][candidate_idx]),
                        "retriever_score": float(score),
                    }
                )
    return pd.DataFrame(rows)


def save_dictionary_pretrain_retriever_cache(cache: Dict[str, Dict], output_dir, *, stem: str) -> Dict[str, str]:
    """Save a retriever cache and a TSV preview."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pickle_path = output_dir / f"{stem}.pkl"
    preview_path = output_dir / f"{stem}.tsv"

    with pickle_path.open("wb") as f:
        pickle.dump(cache, f)
    flatten_dictionary_pretrain_retriever_cache(cache).to_csv(preview_path, sep="\t", index=False)

    return {
        "pickle_path": str(pickle_path),
        "preview_path": str(preview_path),
    }


def load_dictionary_pretrain_retriever_cache(cache_path) -> Dict[str, Dict]:
    """Load a pickled dictionary pretraining retriever cache."""
    with Path(cache_path).open("rb") as f:
        return pickle.load(f)


def infer_dictionary_pretrain_cache_topk(cache: Dict[str, Dict]) -> int:
    """Infer the top-k width stored in a retriever cache."""
    topk_values = []
    for payload in cache.values():
        candidate_scores = payload.get("candidate_scores")
        if candidate_scores is None:
            continue
        if hasattr(candidate_scores, "shape") and len(candidate_scores.shape) == 2:
            topk_values.append(int(candidate_scores.shape[1]))
    return max(topk_values) if topk_values else 0


def trim_dictionary_pretrain_retriever_cache(
    cache: Dict[str, Dict],
    *,
    topk: int,
) -> Dict[str, Dict]:
    """Trim cached candidate arrays to a smaller top-k."""
    requested_topk = max(int(topk), 0)
    trimmed_cache: Dict[str, Dict] = {}
    for entity_type, payload in cache.items():
        trimmed_payload = dict(payload)
        candidate_scores = payload.get("candidate_scores")
        candidate_indices = payload.get("candidate_indices")
        if candidate_scores is not None:
            trimmed_payload["candidate_scores"] = candidate_scores[:, :requested_topk].copy()
        if candidate_indices is not None:
            trimmed_payload["candidate_indices"] = candidate_indices[:, :requested_topk].copy()
        trimmed_cache[entity_type] = trimmed_payload
    return trimmed_cache
