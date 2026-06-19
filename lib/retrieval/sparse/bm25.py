"""BM25 sparse retrieval index."""

import logging
import math
import re
from collections import Counter
from typing import Dict, List

import numpy as np
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


def bm25_tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 retrieval."""
    return re.findall(r"\w+", str(text), flags=re.UNICODE)


class BM25Index:
    """BM25 retrieval index over vocabulary names."""
    def __init__(self, documents: np.ndarray, k1: float, b: float, doc_batch_size: int, show_progress: bool = False):
        self.documents = documents
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_batch_size = int(doc_batch_size)
        self.num_docs = len(documents)
        self.avgdl = 0.0
        self.idf: Dict[str, float] = {}
        logger.info(
            "Initializing BM25Index: num_docs=%d, k1=%.4f, b=%.4f, doc_batch_size=%d",
            self.num_docs,
            self.k1,
            self.b,
            self.doc_batch_size,
        )
        self._build_statistics(show_progress=show_progress)

    def _build_statistics(self, show_progress: bool = False):
        logger.info("Building BM25 statistics for %d documents", self.num_docs)
        with log_timed(logger, "BM25 statistics build"):
            doc_freq = Counter()
            total_doc_length = 0

            iterator = range(0, self.num_docs, self.doc_batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="BM25 stats", unit="v-batch")

            for start in iterator:
                end = min(start + self.doc_batch_size, self.num_docs)
                for doc in self.documents[start:end]:
                    tokens = bm25_tokenize(doc)
                    total_doc_length += len(tokens)
                    for term in set(tokens):
                        doc_freq[term] += 1

            self.avgdl = total_doc_length / max(self.num_docs, 1)
            self.idf = {
                term: math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))
                for term, df in doc_freq.items()
            }
        logger.info(
            "Built BM25 statistics: vocab_terms=%d, avgdl=%.4f",
            len(self.idf),
            self.avgdl,
        )

    def _score_query_terms(self, query_terms, doc_tf: Counter, norm: float) -> float:
        score = 0.0
        for term in query_terms:
            tf = doc_tf.get(term, 0)
            if tf == 0:
                continue
            term_idf = self.idf.get(term, 0.0)
            score += term_idf * (tf * (self.k1 + 1.0)) / (tf + norm)
        return float(score)

    def score_candidates_batched(self, query_texts, candidate_ids_per_query: List[List[int]]) -> List[Dict[int, float]]:
        if len(query_texts) != len(candidate_ids_per_query):
            raise ValueError("query_texts and candidate_ids_per_query must have the same length.")

        non_empty_query_positions = [
            query_idx for query_idx, candidate_ids in enumerate(candidate_ids_per_query) if len(candidate_ids) > 0
        ]
        if len(non_empty_query_positions) == 0:
            return [{} for _ in query_texts]

        logger.debug("Scoring missing BM25 candidates in batch for %d queries", len(non_empty_query_positions))

        query_terms_list = [
            list(set(bm25_tokenize(query_texts[query_idx])))
            for query_idx in non_empty_query_positions
        ]
        avgdl = max(self.avgdl, 1e-12)

        unique_candidate_ids = list(
            iter_unique_ordered(
                candidate_id
                for query_idx in non_empty_query_positions
                for candidate_id in candidate_ids_per_query[query_idx]
            )
        )

        candidate_cache = {}
        for candidate_id in unique_candidate_ids:
            doc_tokens = bm25_tokenize(self.documents[candidate_id])
            doc_tf = Counter(doc_tokens)
            doc_len = len(doc_tokens)
            norm = self.k1 * (1.0 - self.b + self.b * doc_len / avgdl)
            candidate_cache[candidate_id] = (doc_tf, norm)

        result = [{} for _ in query_texts]
        for query_pos, query_idx in enumerate(non_empty_query_positions):
            query_terms = query_terms_list[query_pos]
            scores = {}
            for candidate_id in candidate_ids_per_query[query_idx]:
                doc_tf, norm = candidate_cache[candidate_id]
                scores[candidate_id] = self._score_query_terms(query_terms, doc_tf, norm)
            result[query_idx] = scores

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

        logger.info(
            "Starting BM25 retrieval: num_queries=%d, num_docs=%d, base_k=%d, retrieval_k=%d, query_batch_size=%d, vocab_batch_size=%d, deduplicate_by_cui=%s",
            len(query_names),
            len(self.documents),
            base_k,
            retrieval_k,
            query_batch_size,
            vocab_batch_size,
            deduplicate_by_cui,
        )
        all_scores = []
        all_indices = []

        with log_timed(logger, "BM25 retrieval"):
            q_iterator = range(0, len(query_names), query_batch_size)
            if show_progress:
                q_iterator = tqdm(q_iterator, desc="BM25 retrieval", unit="q-batch")

            avgdl = max(self.avgdl, 1e-12)

            for q_start in q_iterator:
                q_end = min(q_start + query_batch_size, len(query_names))
                logger.debug("BM25 retrieval query batch: start=%d, end=%d", q_start, q_end)
                query_batch = (
                    query_names[q_start:q_end].tolist()
                    if isinstance(query_names, np.ndarray)
                    else query_names[q_start:q_end]
                )
                query_terms_list = [list(set(bm25_tokenize(q))) for q in query_batch]

                batch_best_scores = None
                batch_best_indices = None

                vocab_iterator = range(0, len(self.documents), vocab_batch_size)
                if show_progress:
                    vocab_iterator = tqdm(
                        vocab_iterator,
                        desc=f"BM25 vocab batches [{q_start}:{q_end}]",
                        unit="v-batch",
                        leave=False,
                    )

                for v_start in vocab_iterator:
                    v_end = min(v_start + vocab_batch_size, len(self.documents))
                    logger.debug("BM25 retrieval vocab batch: start=%d, end=%d", v_start, v_end)
                    vocab_batch = (
                        self.documents[v_start:v_end].tolist()
                        if isinstance(self.documents, np.ndarray)
                        else self.documents[v_start:v_end]
                    )
                    local_scores = np.zeros((len(query_batch), len(vocab_batch)), dtype=np.float32)

                    for doc_local_id, doc_text in enumerate(vocab_batch):
                        doc_tokens = bm25_tokenize(doc_text)
                        doc_tf = Counter(doc_tokens)
                        doc_len = len(doc_tokens)
                        norm = self.k1 * (1.0 - self.b + self.b * doc_len / avgdl)

                        for query_local_id, query_terms in enumerate(query_terms_list):
                            local_scores[query_local_id, doc_local_id] = self._score_query_terms(
                                query_terms,
                                doc_tf,
                                norm,
                            )

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
            "BM25 retrieval finished: score_shape=%s, index_shape=%s",
            result_scores.shape,
            result_indices.shape,
        )
        return result_scores, result_indices
