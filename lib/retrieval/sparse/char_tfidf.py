"""Character TF-IDF sparse retrieval index."""

import logging
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from ..core.batch_utils import (
    deduplicate_topk_by_key,
    iter_unique_ordered,
    merge_topk_arrays,
    resolve_retrieval_k,
    topk_from_2d_scores,
)
from lib.utils.logging_utils import log_timed

logger = logging.getLogger(__name__)


class CharTfidfIndex:
    """Character n-gram TF-IDF retrieval index."""
    def __init__(
        self,
        documents: np.ndarray,
        min_ngram: int,
        max_ngram: int,
        analyzer: str = "char",
        lowercase: bool = False,
        show_progress: bool = False,
    ):
        self.documents = documents
        self.min_ngram = int(min_ngram)
        self.max_ngram = int(max_ngram)
        self.analyzer = analyzer
        self.lowercase = bool(lowercase)
        self.num_docs = len(documents)
        self.show_progress = bool(show_progress)

        logger.info(
            "Initializing CharTfidfIndex: num_docs=%d, analyzer=%s, ngram_range=(%d, %d), lowercase=%s, show_progress=%s",
            self.num_docs,
            self.analyzer,
            self.min_ngram,
            self.max_ngram,
            self.lowercase,
            self.show_progress,
        )
        self.vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=(self.min_ngram, self.max_ngram),
            lowercase=self.lowercase,
            dtype=np.float32,
            norm="l2",
        )
        with log_timed(logger, "Char TF-IDF fit"):
            self.vectorizer.fit(self.documents.tolist())
        logger.info(
            "Built CharTfidfIndex: vocab_terms=%d",
            len(self.vectorizer.vocabulary_),
        )

    def score_candidates_batched(self, query_texts, candidate_ids_per_query: List[List[int]]) -> List[Dict[int, float]]:
        if len(query_texts) != len(candidate_ids_per_query):
            raise ValueError("query_texts and candidate_ids_per_query must have the same length.")

        non_empty_query_positions = [
            query_idx for query_idx, candidate_ids in enumerate(candidate_ids_per_query) if len(candidate_ids) > 0
        ]
        if len(non_empty_query_positions) == 0:
            return [{} for _ in query_texts]

        logger.debug("Scoring missing char TF-IDF candidates in batch for %d queries", len(non_empty_query_positions))

        active_query_texts = [query_texts[query_idx] for query_idx in non_empty_query_positions]
        query_matrix = self.vectorizer.transform(active_query_texts)

        unique_candidate_ids = list(
            iter_unique_ordered(
                candidate_id
                for query_idx in non_empty_query_positions
                for candidate_id in candidate_ids_per_query[query_idx]
            )
        )

        candidate_documents = self.documents[unique_candidate_ids].tolist()
        candidate_matrix = self.vectorizer.transform(candidate_documents)
        score_matrix = (query_matrix @ candidate_matrix.T).toarray().astype(np.float32)
        candidate_id2position = {candidate_id: pos for pos, candidate_id in enumerate(unique_candidate_ids)}

        result = [{} for _ in query_texts]
        for query_pos, query_idx in enumerate(non_empty_query_positions):
            result[query_idx] = {
                candidate_id: float(score_matrix[query_pos, candidate_id2position[candidate_id]])
                for candidate_id in candidate_ids_per_query[query_idx]
            }

        return result

    def batch_score_topk(
        self,
        query_names,
        base_k: int,
        query_batch_size: int,
        vocab_batch_size: int,
        vocab_cuis=None,
        deduplicate_by_cui: bool = False,
        cui_overfetch_factor: int = 4,
        show_progress: bool = False,
    ):
        retrieval_k = resolve_retrieval_k(
            num_items=len(self.documents),
            base_k=base_k,
            deduplicate_by_cui=deduplicate_by_cui,
            vocab_cuis=vocab_cuis,
            cui_overfetch_factor=cui_overfetch_factor,
        )
        effective_vocab_batch_size = len(self.documents) if vocab_batch_size <= 0 else int(vocab_batch_size)

        logger.info(
            "Starting char TF-IDF retrieval: num_queries=%d, num_docs=%d, base_k=%d, retrieval_k=%d, query_batch_size=%d, vocab_batch_size=%d, deduplicate_by_cui=%s",
            len(query_names),
            len(self.documents),
            base_k,
            retrieval_k,
            query_batch_size,
            effective_vocab_batch_size,
            deduplicate_by_cui,
        )
        all_scores = []
        all_indices = []

        with log_timed(logger, "Char TF-IDF retrieval"):
            q_iterator = range(0, len(query_names), query_batch_size)
            if show_progress:
                q_iterator = tqdm(q_iterator, desc="Char TF-IDF retrieval", unit="q-batch")

            for q_start in q_iterator:
                q_end = min(q_start + query_batch_size, len(query_names))
                logger.debug("Char TF-IDF query batch: start=%d, end=%d", q_start, q_end)
                query_batch = (
                    query_names[q_start:q_end].tolist()
                    if isinstance(query_names, np.ndarray)
                    else query_names[q_start:q_end]
                )
                query_matrix = self.vectorizer.transform(query_batch)
                batch_scores = np.full((len(query_batch), retrieval_k), float("-inf"), dtype=np.float32)
                batch_indices = np.full((len(query_batch), retrieval_k), -1, dtype=np.int64)

                v_iterator = range(0, len(self.documents), effective_vocab_batch_size)
                if show_progress:
                    v_iterator = tqdm(
                        v_iterator,
                        desc=f"Char TF-IDF vocab batches [{q_start}:{q_end}]",
                        unit="v-batch",
                        leave=False,
                    )

                for v_start in v_iterator:
                    v_end = min(v_start + effective_vocab_batch_size, len(self.documents))
                    logger.debug("Char TF-IDF vocab batch: start=%d, end=%d", v_start, v_end)
                    vocab_batch_docs = self.documents[v_start:v_end].tolist()
                    vocab_batch_matrix = self.vectorizer.transform(vocab_batch_docs)
                    score_matrix = (query_matrix @ vocab_batch_matrix.T).toarray().astype(np.float32)

                    local_scores, local_indices = topk_from_2d_scores(score_matrix, topk=retrieval_k)
                    local_indices = local_indices + v_start
                    batch_scores, batch_indices = merge_topk_arrays(
                        prev_scores=batch_scores,
                        prev_indices=batch_indices,
                        new_scores=local_scores,
                        new_indices=local_indices,
                        topk=retrieval_k,
                    )

                all_scores.append(batch_scores)
                all_indices.append(batch_indices)

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
            "Char TF-IDF retrieval finished: score_shape=%s, index_shape=%s",
            result_scores.shape,
            result_indices.shape,
        )
        return result_scores, result_indices
