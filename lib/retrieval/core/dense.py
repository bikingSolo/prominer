"""Dense embedding retrieval utilities."""

import logging
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from .batch_utils import (
    deduplicate_topk_by_key,
    iter_unique_ordered,
    merge_topk_arrays,
    resolve_retrieval_k,
    topk_from_2d_scores,
)
from lib.utils.logging_utils import log_timed

logger = logging.getLogger(__name__)


def encode_names(
    names,
    st_model: SentenceTransformer,
    batch_size: int = 256,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode names with a sentence-transformer model."""
    if isinstance(names, np.ndarray):
        names = names.tolist()

    logger.debug("Encoding %d texts with batch_size=%d", len(names), batch_size)

    with log_timed(logger, f"Encoding {len(names)} texts", level=logging.DEBUG):
        embeddings = st_model.encode(
            names,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    return embeddings.astype(np.float32)


def get_dense_topk_batched(
    query_names,
    vocab_names,
    st_model: SentenceTransformer,
    base_k: int,
    query_batch_size: int,
    vocab_batch_size: int,
    st_encode_batch_size: int,
    vocab_cuis=None,
    deduplicate_by_cui: bool = False,
    cui_overfetch_factor: int = 4,
    show_progress: bool = False,
):
    """Retrieve dense top-k candidates in batches."""
    retrieval_k = resolve_retrieval_k(
        num_items=len(vocab_names),
        base_k=base_k,
        deduplicate_by_cui=deduplicate_by_cui,
        vocab_cuis=vocab_cuis,
        cui_overfetch_factor=cui_overfetch_factor,
    )

    logger.info(
        "Starting dense retrieval: num_queries=%d, num_vocab=%d, base_k=%d, retrieval_k=%d, query_batch_size=%d, vocab_batch_size=%d, deduplicate_by_cui=%s",
        len(query_names),
        len(vocab_names),
        base_k,
        retrieval_k,
        query_batch_size,
        vocab_batch_size,
        deduplicate_by_cui,
    )
    all_scores = []
    all_indices = []

    with log_timed(logger, "Dense retrieval"):
        q_iterator = range(0, len(query_names), query_batch_size)
        if show_progress:
            q_iterator = tqdm(q_iterator, desc="Dense retrieval", unit="q-batch")

        for q_start in q_iterator:
            q_end = min(q_start + query_batch_size, len(query_names))
            logger.debug("Dense retrieval query batch: start=%d, end=%d", q_start, q_end)
            query_batch_names = (
                query_names[q_start:q_end].tolist()
                if isinstance(query_names, np.ndarray)
                else query_names[q_start:q_end]
            )

            query_embeds = encode_names(
                names=query_batch_names,
                st_model=st_model,
                batch_size=min(st_encode_batch_size, len(query_batch_names)) or 1,
                show_progress=False,
            )

            batch_best_scores = None
            batch_best_indices = None

            vocab_iterator = range(0, len(vocab_names), vocab_batch_size)
            if show_progress:
                vocab_iterator = tqdm(
                    vocab_iterator,
                    desc=f"Dense vocab batches [{q_start}:{q_end}]",
                    unit="v-batch",
                    leave=False,
                )

            for v_start in vocab_iterator:
                v_end = min(v_start + vocab_batch_size, len(vocab_names))
                logger.debug("Dense retrieval vocab batch: start=%d, end=%d", v_start, v_end)
                vocab_batch_names = (
                    vocab_names[v_start:v_end].tolist()
                    if isinstance(vocab_names, np.ndarray)
                    else vocab_names[v_start:v_end]
                )

                vocab_embeds = encode_names(
                    names=vocab_batch_names,
                    st_model=st_model,
                    batch_size=min(st_encode_batch_size, len(vocab_batch_names)) or 1,
                    show_progress=False,
                )

                local_scores = np.matmul(query_embeds, vocab_embeds.T).astype(np.float32)
                local_top_scores, local_top_indices = topk_from_2d_scores(local_scores, topk=retrieval_k)
                local_top_indices = local_top_indices + v_start

                if batch_best_scores is None:
                    batch_best_scores = local_top_scores
                    batch_best_indices = local_top_indices
                else:
                    batch_best_scores, batch_best_indices = merge_topk_arrays(
                        batch_best_scores,
                        batch_best_indices,
                        local_top_scores,
                        local_top_indices,
                        topk=retrieval_k,
                    )

            all_scores.append(batch_best_scores)
            all_indices.append(batch_best_indices)

    result_scores = np.concatenate(all_scores, axis=0)
    result_indices = np.concatenate(all_indices, axis=0)
    if deduplicate_by_cui:
        result_scores, result_indices = deduplicate_topk_by_key(
            scores=result_scores,
            indices=result_indices,
            item_keys=vocab_cuis,
            topk=base_k,
        )

    logger.info(
        "Dense retrieval finished: score_shape=%s, index_shape=%s",
        result_scores.shape,
        result_indices.shape,
    )
    return result_scores, result_indices


def compute_missing_dense_scores_batched(
    query_texts,
    missing_candidate_ids_per_query: List[List[int]],
    vocab_names: np.ndarray,
    st_model: SentenceTransformer,
    st_encode_batch_size: int,
) -> List[Dict[int, float]]:
    """Compute dense scores for missing query-candidate pairs."""
    if len(query_texts) != len(missing_candidate_ids_per_query):
        raise ValueError("query_texts and missing_candidate_ids_per_query must have the same length.")

    non_empty_query_positions = [
        query_idx for query_idx, candidate_ids in enumerate(missing_candidate_ids_per_query) if len(candidate_ids) > 0
    ]
    if len(non_empty_query_positions) == 0:
        return [{} for _ in query_texts]

    logger.debug(
        "Computing missing dense scores in batch for %d queries",
        len(non_empty_query_positions),
    )

    active_query_texts = [query_texts[query_idx] for query_idx in non_empty_query_positions]
    query_embeds = encode_names(
        names=active_query_texts,
        st_model=st_model,
        batch_size=min(st_encode_batch_size, len(active_query_texts)) or 1,
        show_progress=False,
    )

    unique_candidate_ids = list(
        iter_unique_ordered(
            candidate_id
            for query_idx in non_empty_query_positions
            for candidate_id in missing_candidate_ids_per_query[query_idx]
        )
    )

    candidate_names = [vocab_names[idx] for idx in unique_candidate_ids]
    candidate_embeds = encode_names(
        names=candidate_names,
        st_model=st_model,
        batch_size=min(st_encode_batch_size, len(candidate_names)) or 1,
        show_progress=False,
    )
    candidate_id2position = {candidate_id: pos for pos, candidate_id in enumerate(unique_candidate_ids)}

    pair_query_positions = []
    pair_candidate_positions = []
    pair_candidate_ids = []
    query_id2slice = []
    cursor = 0

    for query_pos, query_idx in enumerate(non_empty_query_positions):
        candidate_ids = missing_candidate_ids_per_query[query_idx]
        for candidate_id in candidate_ids:
            pair_query_positions.append(query_pos)
            pair_candidate_positions.append(candidate_id2position[candidate_id])
            pair_candidate_ids.append(candidate_id)
        next_cursor = cursor + len(candidate_ids)
        query_id2slice.append((query_idx, cursor, next_cursor))
        cursor = next_cursor

    pair_query_positions = np.asarray(pair_query_positions, dtype=np.int64)
    pair_candidate_positions = np.asarray(pair_candidate_positions, dtype=np.int64)
    pair_scores = np.sum(
        query_embeds[pair_query_positions] * candidate_embeds[pair_candidate_positions],
        axis=1,
        dtype=np.float32,
    )

    result = [{} for _ in query_texts]
    for query_idx, start, end in query_id2slice:
        result[query_idx] = {
            candidate_id: float(score)
            for candidate_id, score in zip(pair_candidate_ids[start:end], pair_scores[start:end])
        }

    return result
