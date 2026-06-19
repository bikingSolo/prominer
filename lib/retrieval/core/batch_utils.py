"""Batch-oriented helpers for top-k retrieval arrays."""

from typing import Tuple

import numpy as np


def resolve_retrieval_k(
    *,
    num_items: int,
    base_k: int,
    deduplicate_by_cui: bool,
    vocab_cuis=None,
    cui_overfetch_factor: int = 4,
) -> int:
    """Resolve the retrieval depth used for top-k arrays."""
    if not deduplicate_by_cui:
        return base_k
    if vocab_cuis is None:
        raise ValueError("vocab_cuis must be provided when deduplicate_by_cui=True.")
    return min(num_items, max(base_k, int(base_k * max(cui_overfetch_factor, 1))))


def iter_unique_ordered(items):
    """Yield unique values while preserving their first-seen order."""
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        yield item


def topk_from_2d_scores(scores: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract top-k scores and indices from a two-dimensional score matrix."""
    num_rows, num_cols = scores.shape
    topk = min(topk, num_cols)
    if topk == 0:
        return np.zeros((num_rows, 0), dtype=np.float32), np.zeros((num_rows, 0), dtype=np.int64)

    rows = np.arange(num_rows)[:, None]
    if topk == num_cols:
        topk_indices = np.argsort(-scores, axis=1)
    else:
        topk_indices = np.argpartition(scores, -topk, axis=1)[:, -topk:]
        topk_values = scores[rows, topk_indices]
        order = np.argsort(-topk_values, axis=1)
        topk_indices = np.take_along_axis(topk_indices, order, axis=1)

    topk_scores = scores[rows, topk_indices]
    return topk_scores.astype(np.float32), topk_indices.astype(np.int64)


def merge_topk_arrays(
    prev_scores: np.ndarray,
    prev_indices: np.ndarray,
    new_scores: np.ndarray,
    new_indices: np.ndarray,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge multiple top-k score/index arrays into one ranked array."""
    merged_scores = np.concatenate([prev_scores, new_scores], axis=1)
    merged_indices = np.concatenate([prev_indices, new_indices], axis=1)
    keep_scores, keep_pos = topk_from_2d_scores(merged_scores, topk=topk)
    keep_indices = np.take_along_axis(merged_indices, keep_pos, axis=1)
    return keep_scores, keep_indices


def deduplicate_topk_by_key(
    scores: np.ndarray,
    indices: np.ndarray,
    item_keys,
    topk: int,
    pad_index: int = -1,
    pad_score: float = float("-inf"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Deduplicate top-k candidates by an external candidate key."""
    if scores.shape != indices.shape:
        raise ValueError("scores and indices must have the same shape.")

    item_keys = np.asarray(item_keys)
    if len(item_keys) == 0:
        return np.zeros((scores.shape[0], 0), dtype=np.float32), np.zeros((scores.shape[0], 0), dtype=np.int64)

    result_width = min(topk, len(set(item_keys.tolist())))
    if result_width == 0:
        return np.zeros((scores.shape[0], 0), dtype=np.float32), np.zeros((scores.shape[0], 0), dtype=np.int64)

    dedup_scores = np.full((scores.shape[0], result_width), pad_score, dtype=np.float32)
    dedup_indices = np.full((indices.shape[0], result_width), pad_index, dtype=np.int64)

    for row_id in range(scores.shape[0]):
        seen_keys = set()
        write_pos = 0

        for score, idx in zip(scores[row_id], indices[row_id]):
            idx = int(idx)
            if idx < 0:
                continue

            item_key = item_keys[idx]
            if item_key in seen_keys:
                continue

            seen_keys.add(item_key)
            dedup_scores[row_id, write_pos] = float(score)
            dedup_indices[row_id, write_pos] = idx
            write_pos += 1

            if write_pos == result_width:
                break

    return dedup_scores, dedup_indices
