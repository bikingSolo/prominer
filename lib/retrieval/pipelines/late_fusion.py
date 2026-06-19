"""Late-fusion retrieval prediction pipeline."""

import logging
from typing import Dict

import pandas as pd
from tqdm.auto import tqdm

from ..core.dense import get_dense_topk_batched
from ..core.hybrid import fuse_dense_and_sparse
from ..sparse.base import build_sparse_index
from ..tuning import build_predictions_dataframe
from lib.utils.logging_utils import log_timed

logger = logging.getLogger(__name__)


def get_type_resource_with_sparse(
    vocab_df: pd.DataFrame,
    entity_type: str,
    resource_cache: Dict[str, Dict],
    sparse_type: str,
    sparse_index_params: Dict,
):
    """Prepare dense and sparse resources for one entity type."""
    if entity_type in resource_cache:
        logger.info("Using cached resources for entity_type=%s", entity_type)
        return resource_cache[entity_type]

    logger.info(
        "Building resources for entity_type=%s with sparse_type=%s and sparse_index_params=%s",
        entity_type,
        sparse_type,
        sparse_index_params,
    )
    subset_vocab = vocab_df[vocab_df["semantic_type"] == entity_type].reset_index(drop=True)
    vocab_names = subset_vocab["concept_name"].astype(str).values
    vocab_cuis = subset_vocab["CUI"].astype(str).values

    sparse_index = build_sparse_index(
        sparse_type=sparse_type,
        documents=vocab_names,
        sparse_index_params={**sparse_index_params, "show_progress": True},
    )

    resource_cache[entity_type] = {
        "vocab_names": vocab_names,
        "vocab_cuis": vocab_cuis,
        "sparse_index": sparse_index,
    }
    logger.info("Built resources for entity_type=%s: vocab_size=%d", entity_type, len(vocab_names))
    return resource_cache[entity_type]


def make_hybrid_predictions_with_sparse(
    entities_df,
    vocab_df,
    resource_cache,
    st_model,
    topk: int,
    candidate_pool_size: int,
    query_batch_size: int,
    dense_vocab_batch_size: int,
    sparse_query_batch_size: int,
    sparse_vocab_batch_size: int,
    dense_weight: float,
    sparse_weight: float,
    sparse_type: str,
    sparse_index_params: Dict,
    st_encode_batch_size: int,
    deduplicate_by_cui: bool = True,
    cui_overfetch_factor: int = 4,
):
    """Generate hybrid dense-sparse predictions for entity mentions."""
    logger.info(
        "Starting hybrid prediction pipeline: num_entities=%d, num_vocab=%d, topk=%d, candidate_pool_size=%d",
        len(entities_df),
        len(vocab_df),
        topk,
        candidate_pool_size,
    )
    prediction_frames = []
    entity_types = sorted(entities_df["entity_type"].dropna().unique().tolist())

    with log_timed(logger, "Hybrid prediction pipeline"):
        entity_iterator = tqdm(entity_types, desc="Entity types", unit="type")
        for entity_type in entity_iterator:
            subset_df = entities_df[entities_df["entity_type"] == entity_type].reset_index(drop=True)
            logger.info("Processing entity_type=%s with %d queries", entity_type, len(subset_df))
            resource = get_type_resource_with_sparse(
                vocab_df=vocab_df,
                entity_type=entity_type,
                resource_cache=resource_cache,
                sparse_type=sparse_type,
                sparse_index_params=sparse_index_params,
            )

            vocab_names = resource["vocab_names"]
            vocab_cuis = resource["vocab_cuis"]
            sparse_index = resource["sparse_index"]

            query_names = subset_df["text"].astype(str).values
            document_ids = subset_df["document_id"].values
            spans = subset_df["spans"].values

            dense_scores, dense_indices = get_dense_topk_batched(
                query_names=query_names,
                vocab_names=vocab_names,
                vocab_cuis=vocab_cuis,
                st_model=st_model,
                base_k=candidate_pool_size,
                query_batch_size=query_batch_size,
                vocab_batch_size=dense_vocab_batch_size,
                st_encode_batch_size=st_encode_batch_size,
                deduplicate_by_cui=deduplicate_by_cui,
                cui_overfetch_factor=cui_overfetch_factor,
                show_progress=True,
            )

            sparse_scores, sparse_indices = sparse_index.batch_score_topk(
                query_names=query_names,
                base_k=candidate_pool_size,
                query_batch_size=sparse_query_batch_size,
                vocab_batch_size=sparse_vocab_batch_size,
                vocab_cuis=vocab_cuis,
                deduplicate_by_cui=deduplicate_by_cui,
                cui_overfetch_factor=cui_overfetch_factor,
                show_progress=True,
            )

            hybrid_scores, hybrid_indices = fuse_dense_and_sparse(
                query_names=query_names,
                vocab_names=vocab_names,
                vocab_cuis=vocab_cuis,
                sparse_index=sparse_index,
                st_model=st_model,
                st_encode_batch_size=st_encode_batch_size,
                dense_scores=dense_scores,
                dense_indices=dense_indices,
                sparse_scores=sparse_scores,
                sparse_indices=sparse_indices,
                topk=topk,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )

            prediction_frames.append(
                build_predictions_dataframe(
                    document_ids=document_ids,
                    spans=spans,
                    vocab_cuis=vocab_cuis,
                    hybrid_indices=hybrid_indices,
                    hybrid_scores=hybrid_scores,
                )
            )

    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    logger.info("Hybrid prediction pipeline finished: num_prediction_rows=%d", len(predictions_df))
    return predictions_df
