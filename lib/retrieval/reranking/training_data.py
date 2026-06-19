"""Training data builders for cross-encoder reranking."""

import logging
import random
from typing import Dict

import numpy as np
import pandas as pd
from datasets import Dataset as HFDataset

from .candidate_cache import (
    _build_gold_text_lookup,
    _build_query_primary_key,
    _ensure_candidate_text_map,
    _resolve_candidate_text,
)


logger = logging.getLogger(__name__)


def build_cross_encoder_training_data(
    train_df: pd.DataFrame,
    retriever_cache: Dict[str, Dict],
    vocab_df: pd.DataFrame,
    *,
    mention_column: str = "text",
    candidate_text_map: Dict[tuple[str, str], str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build cross-encoder training rows from candidate rankings."""
    candidate_text_map = _ensure_candidate_text_map(candidate_text_map)
    gold_text_lookup = _build_gold_text_lookup(vocab_df=vocab_df, candidate_text_map=candidate_text_map)
    rng = random.Random(int(seed))

    required_columns = {"document_id", "spans", "entity_type", "UMLS_CUI", mention_column}
    missing_columns = required_columns.difference(train_df.columns)
    if missing_columns:
        raise ValueError(f"train_df is missing required columns: {sorted(missing_columns)}")

    query_lookup = {
        _build_query_primary_key(row["document_id"], row["spans"]): row
        for _, row in train_df.iterrows()
        if str(row["UMLS_CUI"]) != "CUILESS"
    }

    rows = []
    added_fallback_gold = 0
    for entity_type, payload in retriever_cache.items():
        for query_idx, (document_id, spans, candidate_scores, candidate_indices) in enumerate(
            zip(payload["document_ids"], payload["spans"], payload["candidate_scores"], payload["candidate_indices"])
        ):
            query_pk = _build_query_primary_key(document_id, spans)
            query_row = query_lookup.get(query_pk)
            if query_row is None:
                continue

            gold_cui = str(query_row["UMLS_CUI"])
            docs = []
            labels = []
            seen_candidate_cuis = set()

            for score, candidate_idx in zip(candidate_scores.tolist(), candidate_indices.tolist()):
                candidate_idx = int(candidate_idx)
                if candidate_idx < 0:
                    continue
                candidate_cui = str(payload["vocab_cuis"][candidate_idx])
                if candidate_cui in seen_candidate_cuis:
                    continue
                seen_candidate_cuis.add(candidate_cui)
                docs.append(
                    _resolve_candidate_text(
                        candidate_cui=candidate_cui,
                        entity_type=entity_type,
                        fallback_name=str(payload["vocab_names"][candidate_idx]),
                        candidate_text_map=candidate_text_map,
                    )
                )
                labels.append(1.0 if candidate_cui == gold_cui else 0.0)

            if not any(label > 0 for label in labels):
                fallback_candidates = gold_text_lookup.get((gold_cui, entity_type), [])
                if fallback_candidates:
                    fallback_text = rng.choice(fallback_candidates)
                    if docs:
                        docs[-1] = fallback_text
                        labels[-1] = 1.0
                    else:
                        docs = [fallback_text]
                        labels = [1.0]
                    added_fallback_gold += 1
                else:
                    logger.warning(
                        "Gold CUI=%s entity_type=%s not found in vocab fallback lookup for query_pk=%s",
                        gold_cui,
                        entity_type,
                        query_pk,
                    )
                    continue

            rows.append(
                {
                    "query": str(query_row[mention_column]),
                    "query_raw": str(query_row.get("text", query_row.get("mention_text", query_row[mention_column]))),
                    "query_full_context": str(query_row[mention_column]),
                    "query_base": str(query_row[mention_column]),
                    "query_has_context": bool(
                        str(query_row.get("left_context_text", "")).strip()
                        or str(query_row.get("right_context_text", "")).strip()
                        or str(query_row.get("context_text", "")).strip()
                    ),
                    "query_has_both_context": bool(
                        str(query_row.get("left_context_text", "")).strip()
                        and str(query_row.get("right_context_text", "")).strip()
                    ),
                    "docs": docs,
                    "label": labels,
                    "document_id": str(document_id),
                    "spans": str(spans),
                    "entity_type": str(entity_type),
                    "gold_cui": gold_cui,
                    "query_pk": query_pk,
                }
            )

    train_examples_df = pd.DataFrame(rows)
    if not train_examples_df.empty:
        train_examples_df["row_id"] = np.arange(len(train_examples_df), dtype=np.int64)
    logger.info(
        "Prepared cross-encoder training data: num_rows=%d, fallback_gold_insertions=%d",
        len(train_examples_df),
        added_fallback_gold,
    )
    return train_examples_df


def build_cross_encoder_pairwise_training_data(train_examples_df: pd.DataFrame) -> pd.DataFrame:
    """Build pairwise cross-encoder training examples."""
    rows = []
    for _, row in train_examples_df.iterrows():
        query = str(row["query"])
        docs = list(row["docs"])
        labels = list(row["label"])
        metadata = {
            "document_id": str(row["document_id"]),
            "spans": str(row["spans"]),
            "entity_type": str(row["entity_type"]),
            "gold_cui": str(row["gold_cui"]),
            "query_pk": str(row["query_pk"]),
            "row_id": int(row["row_id"]) if "row_id" in row else -1,
            "query_base": str(row["query_base"]) if "query_base" in row else query,
        }
        for candidate_rank, (doc_text, label) in enumerate(zip(docs, labels), start=1):
            rows.append(
                {
                    "query": query,
                    "candidate_text": str(doc_text),
                    "label": float(label),
                    "candidate_rank": int(candidate_rank),
                    **metadata,
                }
            )
    pairwise_df = pd.DataFrame(rows)
    logger.info(
        "Expanded cross-encoder listwise data to pairwise rows=%d from num_lists=%d",
        len(pairwise_df),
        len(train_examples_df),
    )
    return pairwise_df


def build_cross_encoder_dataset(train_examples_df: pd.DataFrame) -> HFDataset:
    """Convert cross-encoder rows into a Hugging Face dataset."""
    dataset_columns = ["query", "candidate_text", "label"]
    dataset = HFDataset.from_pandas(train_examples_df[dataset_columns], preserve_index=False)
    logger.info("Built CrossEncoder dataset with num_rows=%d", len(dataset))
    return dataset


def build_cross_encoder_listwise_dataset(train_examples_df: pd.DataFrame) -> HFDataset:
    """Convert listwise reranking rows into a Hugging Face dataset."""
    dataset_columns = ["query", "docs", "label"]
    dataset = HFDataset.from_pandas(train_examples_df[dataset_columns], preserve_index=False)
    logger.info("Built CrossEncoder listwise dataset with num_rows=%d", len(dataset))
    return dataset
