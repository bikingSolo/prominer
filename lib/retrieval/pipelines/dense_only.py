"""Dense-only retrieval prediction pipeline."""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from ..core.dense import get_dense_topk_batched
from ..tuning import build_predictions_dataframe, evaluate_dev_predictions
from lib.utils.logging_utils import log_timed

logger = logging.getLogger(__name__)


def get_dense_type_resource(vocab_df: pd.DataFrame, entity_type: str, resource_cache: Dict[str, Dict]):
    """Prepare dense retrieval resources for one entity type."""
    if entity_type in resource_cache:
        return resource_cache[entity_type]

    subset_vocab = vocab_df[vocab_df["semantic_type"] == entity_type].reset_index(drop=True)
    resource_cache[entity_type] = {
        "vocab_names": subset_vocab["concept_name"].astype(str).values,
        "vocab_cuis": subset_vocab["CUI"].astype(str).values,
    }
    logger.info(
        "Prepared dense-only resource for entity_type=%s with vocab_size=%d",
        entity_type,
        len(subset_vocab),
    )
    return resource_cache[entity_type]


def make_dense_predictions(
    data_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    st_model,
    *,
    mention_column: str = "text",
    topk: int,
    query_batch_size: int,
    dense_vocab_batch_size: int,
    st_encode_batch_size: int,
    deduplicate_by_cui: bool,
    resource_cache: Dict[str, Dict] | None = None,
) -> pd.DataFrame:
    """Generate dense-only predictions for entity mentions."""
    if mention_column not in data_df.columns:
        raise ValueError(f"data_df is missing required mention_column={mention_column!r}")
    if resource_cache is None:
        resource_cache = {}

    prediction_frames = []
    entity_types = sorted(data_df["entity_type"].dropna().unique().tolist())

    with log_timed(logger, f"Dense-only prediction for {len(entity_types)} entity types using mention_column={mention_column!r}"):
        for entity_type in entity_types:
            subset_df = data_df[data_df["entity_type"] == entity_type].reset_index(drop=True)
            resource = get_dense_type_resource(vocab_df, entity_type, resource_cache)
            vocab_names = resource["vocab_names"]
            vocab_cuis = resource["vocab_cuis"]

            dense_scores, dense_indices = get_dense_topk_batched(
                query_names=subset_df[mention_column].astype(str).values,
                vocab_names=vocab_names,
                vocab_cuis=vocab_cuis,
                st_model=st_model,
                base_k=topk,
                query_batch_size=query_batch_size,
                vocab_batch_size=dense_vocab_batch_size,
                st_encode_batch_size=st_encode_batch_size,
                deduplicate_by_cui=deduplicate_by_cui,
                show_progress=True,
            )

            prediction_frames.append(
                build_predictions_dataframe(
                    document_ids=subset_df["document_id"].values,
                    spans=subset_df["spans"].values,
                    vocab_cuis=vocab_cuis,
                    hybrid_indices=dense_indices,
                    hybrid_scores=dense_scores,
                )
            )

    if not prediction_frames:
        return pd.DataFrame()
    return pd.concat(prediction_frames, ignore_index=True)


def evaluate_dense_retrieval(
    data_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    st_model,
    *,
    mention_column: str = "text",
    topk: int,
    query_batch_size: int,
    dense_vocab_batch_size: int,
    st_encode_batch_size: int,
    deduplicate_by_cui: bool,
    resource_cache: Dict[str, Dict] | None = None,
):
    """Evaluate dense retrieval predictions against gold CUIs."""
    predictions_df = make_dense_predictions(
        data_df=data_df,
        vocab_df=vocab_df,
        st_model=st_model,
        mention_column=mention_column,
        topk=topk,
        query_batch_size=query_batch_size,
        dense_vocab_batch_size=dense_vocab_batch_size,
        st_encode_batch_size=st_encode_batch_size,
        deduplicate_by_cui=deduplicate_by_cui,
        resource_cache=resource_cache,
    )
    metrics = evaluate_dev_predictions(predictions_df=predictions_df, data_df=data_df)
    return predictions_df, metrics


def predict_dense_to_path(
    data_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    st_model,
    output_path,
    *,
    mention_column: str = "text",
    topk: int,
    query_batch_size: int,
    dense_vocab_batch_size: int,
    st_encode_batch_size: int,
    deduplicate_by_cui: bool,
    resource_cache: Dict[str, Dict] | None = None,
):
    """Write dense-only predictions to a TSV file."""
    predictions_df = make_dense_predictions(
        data_df=data_df,
        vocab_df=vocab_df,
        st_model=st_model,
        mention_column=mention_column,
        topk=topk,
        query_batch_size=query_batch_size,
        dense_vocab_batch_size=dense_vocab_batch_size,
        st_encode_batch_size=st_encode_batch_size,
        deduplicate_by_cui=deduplicate_by_cui,
        resource_cache=resource_cache,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, sep="\t", index=False)
    return predictions_df, output_path
