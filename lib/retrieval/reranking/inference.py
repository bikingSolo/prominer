"""Cross-encoder inference over retriever candidate caches."""

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from typing import Dict

from ..tuning import build_predictions_dataframe
from .candidate_cache import _build_query_primary_key, _ensure_candidate_text_map, _resolve_candidate_text


def _score_pairs_in_batches(
    model: CrossEncoder,
    pairs: list[tuple[str, str]],
    *,
    batch_size: int,
) -> np.ndarray:
    if not pairs:
        return np.asarray([], dtype=np.float32)
    scores = model.predict(
        pairs,
        batch_size=int(batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.asarray(scores, dtype=np.float32).reshape(-1)


def rerank_from_candidate_cache(
    data_df: pd.DataFrame,
    retriever_cache: Dict[str, Dict],
    cross_encoder_model: CrossEncoder,
    *,
    mention_column: str,
    candidate_text_map: Dict[tuple[str, str], str] | None = None,
    batch_size: int = 32,
    topk: int | None = None,
) -> pd.DataFrame:
    """Rerank retriever candidates with a cross-encoder."""
    candidate_text_map = _ensure_candidate_text_map(candidate_text_map)
    query_lookup = {
        _build_query_primary_key(row["document_id"], row["spans"]): row
        for _, row in data_df.iterrows()
    }

    prediction_frames = []
    for entity_type, payload in retriever_cache.items():
        pairs = []
        row_sizes = []
        valid_rows = []
        retriever_scores_kept = []

        for document_id, spans, score_row, index_row in zip(
            payload["document_ids"],
            payload["spans"],
            payload["candidate_scores"],
            payload["candidate_indices"],
        ):
            query_pk = _build_query_primary_key(document_id, spans)
            query_row = query_lookup.get(query_pk)
            if query_row is None:
                continue

            query_text = str(query_row[mention_column])
            candidate_indices = []
            candidate_retriever_scores = []
            candidate_pairs = []
            for score, candidate_idx in zip(score_row.tolist(), index_row.tolist()):
                candidate_idx = int(candidate_idx)
                if candidate_idx < 0:
                    continue
                candidate_indices.append(candidate_idx)
                candidate_retriever_scores.append(float(score))
                candidate_pairs.append(
                    (
                        query_text,
                        _resolve_candidate_text(
                            candidate_cui=str(payload["vocab_cuis"][candidate_idx]),
                            entity_type=entity_type,
                            fallback_name=str(payload["vocab_names"][candidate_idx]),
                            candidate_text_map=candidate_text_map,
                        ),
                    )
                )

            if not candidate_pairs:
                continue
            valid_rows.append((str(document_id), str(spans), candidate_indices))
            retriever_scores_kept.append(candidate_retriever_scores)
            row_sizes.append(len(candidate_pairs))
            pairs.extend(candidate_pairs)

        flat_scores = _score_pairs_in_batches(cross_encoder_model, pairs, batch_size=batch_size)
        start = 0
        hybrid_scores = []
        hybrid_indices = []
        document_ids = []
        spans = []

        for (document_id, span_value, candidate_indices), candidate_retriever_scores, row_size in zip(
            valid_rows,
            retriever_scores_kept,
            row_sizes,
        ):
            end = start + row_size
            cross_scores = flat_scores[start:end]
            start = end

            sort_order = np.lexsort(
                (
                    -np.asarray(candidate_retriever_scores, dtype=np.float32),
                    -np.asarray(cross_scores, dtype=np.float32),
                )
            )
            if topk is not None:
                sort_order = sort_order[: int(topk)]

            hybrid_scores.append(np.asarray(cross_scores[sort_order], dtype=np.float32))
            hybrid_indices.append(np.asarray([candidate_indices[idx] for idx in sort_order], dtype=np.int64))
            document_ids.append(document_id)
            spans.append(span_value)

        if hybrid_scores:
            max_width = max(len(row) for row in hybrid_scores)
            padded_scores = np.full((len(hybrid_scores), max_width), -1e16, dtype=np.float32)
            padded_indices = np.full((len(hybrid_indices), max_width), -1, dtype=np.int64)
            for row_idx, (score_row, index_row) in enumerate(zip(hybrid_scores, hybrid_indices)):
                padded_scores[row_idx, : len(score_row)] = score_row
                padded_indices[row_idx, : len(index_row)] = index_row

            prediction_frames.append(
                build_predictions_dataframe(
                    document_ids=document_ids,
                    spans=spans,
                    vocab_cuis=payload["vocab_cuis"],
                    hybrid_indices=padded_indices,
                    hybrid_scores=padded_scores,
                )
            )

    return pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
