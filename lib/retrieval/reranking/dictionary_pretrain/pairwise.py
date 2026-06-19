"""Convert retriever hits into cross-encoder pairwise training rows."""

import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Dict

import pandas as pd

from .constants import PRETRAIN_PAIR_COLUMNS


logger = logging.getLogger(__name__)


def _build_pairwise_rows_for_entity_type(
    entity_type: str,
    payload: Dict,
    query_lookup: Dict[str, Dict],
    candidate_text_map: Dict[tuple[str, str], str],
    num_negatives: int | None,
) -> list[Dict]:
    """Build pairwise positive and negative rows for one entity type."""
    entity_rows = []
    for query_id, score_row, index_row in zip(
        payload["query_ids"],
        payload["candidate_scores"],
        payload["candidate_indices"],
    ):
        query_row = query_lookup.get(str(query_id))
        if query_row is None:
            continue

        gold_cui = str(query_row["CUI"])
        positive_text = str(candidate_text_map.get((gold_cui, entity_type), query_row["candidate_text"]))
        entity_rows.append(
            {
                "query": str(query_row["query_text"]),
                "candidate_text": positive_text,
                "label": 1.0,
                "query_id": str(query_id),
                "query_text": str(query_row["query_text"]),
                "gold_cui": gold_cui,
                "candidate_cui": gold_cui,
                "semantic_type": str(entity_type),
                "split": str(query_row["split"]),
                "candidate_rank": 0,
                "retriever_score": None,
            }
        )

        negative_count = 0
        seen_candidate_cuis = {gold_cui}
        for rank, (score, candidate_idx) in enumerate(zip(score_row.tolist(), index_row.tolist()), start=1):
            candidate_idx = int(candidate_idx)
            if candidate_idx < 0:
                continue

            candidate_cui = str(payload["vocab_cuis"][candidate_idx])
            if candidate_cui in seen_candidate_cuis:
                continue
            seen_candidate_cuis.add(candidate_cui)

            negative_text = str(
                candidate_text_map.get((candidate_cui, entity_type), payload["vocab_names"][candidate_idx])
            )
            entity_rows.append(
                {
                    "query": str(query_row["query_text"]),
                    "candidate_text": negative_text,
                    "label": 0.0,
                    "query_id": str(query_id),
                    "query_text": str(query_row["query_text"]),
                    "gold_cui": gold_cui,
                    "candidate_cui": candidate_cui,
                    "semantic_type": str(entity_type),
                    "split": str(query_row["split"]),
                    "candidate_rank": int(rank),
                    "retriever_score": float(score),
                }
            )
            negative_count += 1
            if num_negatives is not None and negative_count >= int(num_negatives):
                break
    return entity_rows


def build_dictionary_pretrain_pairwise_data(
    queries_df: pd.DataFrame,
    retriever_cache: Dict[str, Dict],
    *,
    candidate_text_map: Dict[tuple[str, str], str],
    num_negatives: int | None = None,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Build positive and negative cross-encoder pairs from retriever candidates."""
    required_columns = {"query_id", "query_text", "CUI", "semantic_type", "split"}
    missing_columns = required_columns.difference(queries_df.columns)
    if missing_columns:
        raise ValueError(f"queries_df is missing required columns: {sorted(missing_columns)}")

    query_lookup = (
        queries_df[["query_id", "query_text", "CUI", "split", "candidate_text"]]
        .astype({"query_id": str, "query_text": str, "CUI": str, "split": str, "candidate_text": str})
        .set_index("query_id")
        .to_dict(orient="index")
    )

    num_workers = max(int(num_workers), 1)
    entity_type_items = list(retriever_cache.items())
    if num_workers > 1 and len(entity_type_items) > 1:
        with ProcessPoolExecutor(max_workers=min(num_workers, len(entity_type_items))) as executor:
            row_batches = executor.map(
                _build_pairwise_rows_for_entity_type,
                [entity_type for entity_type, _ in entity_type_items],
                [payload for _, payload in entity_type_items],
                repeat(query_lookup),
                repeat(candidate_text_map),
                repeat(num_negatives),
            )
            rows = [row for batch in row_batches for row in batch]
    else:
        rows = []
        for entity_type, payload in entity_type_items:
            rows.extend(
                _build_pairwise_rows_for_entity_type(
                    entity_type,
                    payload,
                    query_lookup,
                    candidate_text_map,
                    num_negatives,
                )
            )

    result_df = pd.DataFrame(rows, columns=PRETRAIN_PAIR_COLUMNS)
    logger.info("Prepared dictionary pretrain pairwise rows=%d", len(result_df))
    return result_df
