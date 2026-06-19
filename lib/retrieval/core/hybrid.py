"""Dense and sparse score fusion helpers."""

import logging

import numpy as np

from .batch_utils import iter_unique_ordered
from .dense import compute_missing_dense_scores_batched
from lib.utils.logging_utils import log_timed

logger = logging.getLogger(__name__)


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize a score vector."""
    if len(scores) == 0:
        return scores

    score_min = scores.min()
    score_max = scores.max()
    if score_max - score_min < 1e-12:
        if score_max > 0:
            return np.ones_like(scores, dtype=np.float32)
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - score_min) / (score_max - score_min)).astype(np.float32)


def finalize_prediction_row(
    idx_row,
    score_row,
    vocab_cuis,
):
    """Choose the best fused candidate for one query."""
    valid_pairs = [
        (str(vocab_cuis[int(idx)]), float(score))
        for idx, score in zip(idx_row, score_row)
        if int(idx) >= 0
    ]

    if not valid_pairs:
        return [], []

    final_cuis = [cui for cui, _ in valid_pairs]
    final_scores = [score for _, score in valid_pairs]

    target_len = len(idx_row)
    while len(final_cuis) < target_len:
        final_cuis.append(final_cuis[-1])
        final_scores.append(final_scores[-1])

    return final_cuis, final_scores


def fuse_dense_and_sparse(
    query_names,
    vocab_names: np.ndarray,
    vocab_cuis: np.ndarray,
    sparse_index,
    st_model,
    st_encode_batch_size: int,
    dense_scores: np.ndarray,
    dense_indices: np.ndarray,
    sparse_scores: np.ndarray,
    sparse_indices: np.ndarray,
    topk: int,
    dense_weight: float,
    sparse_weight: float,
):
    """Fuse dense and sparse retrieval scores."""
    logger.info(
        "Starting late fusion: num_queries=%d, topk=%d, dense_weight=%.4f, sparse_weight=%.4f",
        len(dense_indices),
        topk,
        dense_weight,
        sparse_weight,
    )
    final_indices = []
    final_scores = []
    total_union_candidates = 0
    total_missing_dense = 0
    total_missing_sparse = 0

    with log_timed(logger, "Late fusion"):
        candidate_ids_per_query = []
        dense_maps = []
        sparse_maps = []
        missing_dense_ids_per_query = []
        missing_sparse_ids_per_query = []

        for q_id in range(len(dense_indices)):
            dense_row = [int(idx) for idx in dense_indices[q_id].tolist() if int(idx) >= 0]
            sparse_row = [int(idx) for idx in sparse_indices[q_id].tolist() if int(idx) >= 0]
            candidate_ids = list(iter_unique_ordered(dense_row + sparse_row))
            total_union_candidates += len(candidate_ids)

            dense_map = {
                int(idx): float(score)
                for idx, score in zip(dense_indices[q_id], dense_scores[q_id])
                if int(idx) >= 0
            }
            sparse_map = {
                int(idx): float(score)
                for idx, score in zip(sparse_indices[q_id], sparse_scores[q_id])
                if int(idx) >= 0
            }

            missing_dense_ids = [idx for idx in candidate_ids if idx not in dense_map]
            total_missing_dense += len(missing_dense_ids)
            missing_sparse_ids = [idx for idx in candidate_ids if idx not in sparse_map]
            total_missing_sparse += len(missing_sparse_ids)
            candidate_ids_per_query.append(candidate_ids)
            dense_maps.append(dense_map)
            sparse_maps.append(sparse_map)
            missing_dense_ids_per_query.append(missing_dense_ids)
            missing_sparse_ids_per_query.append(missing_sparse_ids)

        batched_missing_dense_scores = compute_missing_dense_scores_batched(
            query_texts=query_names,
            missing_candidate_ids_per_query=missing_dense_ids_per_query,
            vocab_names=vocab_names,
            st_model=st_model,
            st_encode_batch_size=st_encode_batch_size,
        )
        batched_missing_sparse_scores = sparse_index.score_candidates_batched(
            query_texts=query_names,
            candidate_ids_per_query=missing_sparse_ids_per_query,
        )

        for q_id in range(len(dense_indices)):
            candidate_ids = candidate_ids_per_query[q_id]
            dense_map = dense_maps[q_id]
            sparse_map = sparse_maps[q_id]
            if batched_missing_dense_scores[q_id]:
                dense_map.update(batched_missing_dense_scores[q_id])
            if batched_missing_sparse_scores[q_id]:
                sparse_map.update(batched_missing_sparse_scores[q_id])

            dense_candidate_scores = np.array([dense_map[idx] for idx in candidate_ids], dtype=np.float32)
            sparse_candidate_scores = np.array([sparse_map[idx] for idx in candidate_ids], dtype=np.float32)

            dense_norm = minmax_normalize(dense_candidate_scores)
            sparse_norm = minmax_normalize(sparse_candidate_scores)
            hybrid_scores = dense_weight * dense_norm + sparse_weight * sparse_norm

            ranked_positions = np.argsort(-hybrid_scores)
            selected_indices = []
            selected_scores = []
            seen_cuis = set()

            for pos in ranked_positions:
                candidate_id = candidate_ids[int(pos)]
                candidate_cui = str(vocab_cuis[candidate_id])
                if candidate_cui in seen_cuis:
                    continue
                seen_cuis.add(candidate_cui)
                selected_indices.append(candidate_id)
                selected_scores.append(float(hybrid_scores[int(pos)]))
                if len(selected_indices) == topk:
                    break

            while len(selected_indices) < topk:
                selected_indices.append(-1)
                selected_scores.append(float("-inf"))

            final_indices.append(np.asarray(selected_indices, dtype=np.int64))
            final_scores.append(np.asarray(selected_scores, dtype=np.float32))

    score_array = np.stack(final_scores)
    index_array = np.stack(final_indices)
    logger.info(
        "Late fusion finished: avg_union_candidates=%.2f, total_missing_dense=%d, total_missing_sparse=%d, score_shape=%s",
        total_union_candidates / max(len(dense_indices), 1),
        total_missing_dense,
        total_missing_sparse,
        score_array.shape,
    )
    return score_array, index_array
