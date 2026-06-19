"""Late-fusion parameter tuning utilities."""

import logging
from itertools import product
from typing import Dict, Iterable

import pandas as pd
from tqdm.auto import tqdm

from ..core.dense import get_dense_topk_batched
from ..core.evaluation import (
    PREDICTION_COL,
    RANK_COL,
    SAMPLE_PK_COL,
    TRUE_CUI_COL,
    calculate_metrics,
    create_row_primary_key,
    create_sample_pk2_true_cui_map,
)
from ..core.hybrid import finalize_prediction_row, fuse_dense_and_sparse
from ..sparse.base import build_sparse_index
from lib.utils.logging_utils import log_timed

logger = logging.getLogger(__name__)


def build_predictions_dataframe(
    document_ids,
    spans,
    vocab_cuis,
    hybrid_indices,
    hybrid_scores,
) -> pd.DataFrame:
    """Build a prediction dataframe from per-query predictions."""
    logger.debug("Building predictions dataframe for %d queries", len(document_ids))
    rows = []
    for doc_id, sp, idx_row, score_row in zip(document_ids, spans, hybrid_indices, hybrid_scores):
        pred_cuis, pred_scores = finalize_prediction_row(
            idx_row=idx_row,
            score_row=score_row,
            vocab_cuis=vocab_cuis,
        )
        for rank, (pred_cui, hybrid_score) in enumerate(zip(pred_cuis, pred_scores), start=1):
            rows.append(
                {
                    "document_id": doc_id,
                    "spans": sp,
                    "rank": rank,
                    "prediction": pred_cui,
                    "hybrid_score": float(hybrid_score),
                }
            )
    return pd.DataFrame(rows)


def evaluate_dev_predictions(predictions_df: pd.DataFrame, data_df: pd.DataFrame) -> Dict[str, float]:
    """Evaluate development predictions against gold labels."""
    logger.info(
        "Evaluating predictions on dev: num_predictions=%d, num_gold_rows=%d",
        len(predictions_df),
        len(data_df),
    )
    predictions_df = predictions_df.copy()
    predictions_df[SAMPLE_PK_COL] = predictions_df.apply(create_row_primary_key, axis=1)

    eval_df = data_df.copy()
    eval_df[SAMPLE_PK_COL] = eval_df.apply(create_row_primary_key, axis=1)
    eval_df = eval_df[eval_df[TRUE_CUI_COL] != "CUILESS"]

    merged_df = eval_df.merge(
        predictions_df[[SAMPLE_PK_COL, RANK_COL, PREDICTION_COL]],
        on=SAMPLE_PK_COL,
    )
    sample_pk2true_cui = create_sample_pk2_true_cui_map(eval_df, TRUE_CUI_COL)
    logger.info(
        "Prepared dev evaluation frames: num_eval_rows=%d, num_merged_rows=%d",
        len(eval_df),
        len(merged_df),
    )
    return calculate_metrics(merged_df, sample_pk2true_cui)


def prepare_dense_dev_cache(
    entities_df,
    vocab_df,
    st_model,
    candidate_pool_size: int,
    query_batch_size: int,
    dense_vocab_batch_size: int,
    st_encode_batch_size: int,
    deduplicate_by_cui: bool = True,
    cui_overfetch_factor: int = 4,
):
    """Prepare cached dense development retrieval results."""
    logger.info(
        "Preparing dense dev cache: num_entities=%d, num_vocab=%d, candidate_pool_size=%d",
        len(entities_df),
        len(vocab_df),
        candidate_pool_size,
    )
    dense_cache = {}
    entity_types = sorted(entities_df["entity_type"].dropna().unique().tolist())

    with log_timed(logger, "Dense dev cache preparation"):
        entity_iterator = tqdm(entity_types, desc="Dense cache entity types", unit="type")
        for entity_type in entity_iterator:
            subset_df = entities_df[entities_df["entity_type"] == entity_type].reset_index(drop=True)
            subset_vocab = vocab_df[vocab_df["semantic_type"] == entity_type].reset_index(drop=True)
            logger.info(
                "Preparing dense cache for entity_type=%s: num_queries=%d, vocab_size=%d",
                entity_type,
                len(subset_df),
                len(subset_vocab),
            )
            vocab_names = subset_vocab["concept_name"].astype(str).values
            vocab_cuis = subset_vocab["CUI"].astype(str).values

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

            dense_cache[entity_type] = {
                "query_names": query_names,
                "document_ids": document_ids,
                "spans": spans,
                "vocab_names": vocab_names,
                "vocab_cuis": vocab_cuis,
                "dense_scores": dense_scores,
                "dense_indices": dense_indices,
            }

    logger.info("Dense dev cache prepared for %d entity types", len(dense_cache))
    return dense_cache


def make_hybrid_predictions_from_dense_cache_with_sparse(
    dense_cache: Dict[str, Dict],
    topk: int,
    dense_weight: float,
    sparse_weight: float,
    st_model,
    st_encode_batch_size: int,
    sparse_results_cache: Dict[str, Dict],
) -> pd.DataFrame:
    """Build hybrid predictions from dense and sparse caches."""
    logger.info(
        "Building hybrid sparse predictions from dense cache: num_entity_types=%d, topk=%d, dense_weight=%.4f, sparse_weight=%.4f",
        len(dense_cache),
        topk,
        dense_weight,
        sparse_weight,
    )
    prediction_frames = []

    for entity_type, cache in dense_cache.items():
        logger.info("Fusing cached sparse candidates for entity_type=%s", entity_type)
        sparse_index = sparse_results_cache[entity_type]["sparse_index"]
        sparse_scores = sparse_results_cache[entity_type]["sparse_scores"]
        sparse_indices = sparse_results_cache[entity_type]["sparse_indices"]

        hybrid_scores, hybrid_indices = fuse_dense_and_sparse(
            query_names=cache["query_names"],
            vocab_names=cache["vocab_names"],
            vocab_cuis=cache["vocab_cuis"],
            sparse_index=sparse_index,
            st_model=st_model,
            st_encode_batch_size=st_encode_batch_size,
            dense_scores=cache["dense_scores"],
            dense_indices=cache["dense_indices"],
            sparse_scores=sparse_scores,
            sparse_indices=sparse_indices,
            topk=topk,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

        prediction_frames.append(
            build_predictions_dataframe(
                document_ids=cache["document_ids"],
                spans=cache["spans"],
                vocab_cuis=cache["vocab_cuis"],
                hybrid_indices=hybrid_indices,
                hybrid_scores=hybrid_scores,
            )
        )
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    logger.info("Built cached hybrid sparse predictions: num_prediction_rows=%d", len(predictions_df))
    return predictions_df


def prepare_sparse_results_cache(
    dense_cache: Dict[str, Dict],
    sparse_type: str,
    sparse_index_params: Dict,
    candidate_pool_size: int,
    query_batch_size: int,
    sparse_vocab_batch_size: int,
    deduplicate_by_cui: bool = True,
    cui_overfetch_factor: int = 4,
):
    """Prepare cached sparse retrieval results for tuning."""
    logger.info(
        "Preparing sparse cache: num_entity_types=%d, sparse_type=%s, sparse_index_params=%s",
        len(dense_cache),
        sparse_type,
        sparse_index_params,
    )
    sparse_results_cache = {}

    with log_timed(logger, f"Sparse cache preparation ({sparse_type})"):
        entity_iterator = tqdm(dense_cache.items(), desc=f"{sparse_type} cache entity types", unit="type")
        for entity_type, cache in entity_iterator:
            logger.info(
                "Preparing sparse cache for entity_type=%s: num_queries=%d, vocab_size=%d",
                entity_type,
                len(cache["query_names"]),
                len(cache["vocab_names"]),
            )
            sparse_index = build_sparse_index(
                sparse_type=sparse_type,
                documents=cache["vocab_names"],
                sparse_index_params={**sparse_index_params, "show_progress": True},
            )

            sparse_scores, sparse_indices = sparse_index.batch_score_topk(
                query_names=cache["query_names"],
                base_k=candidate_pool_size,
                query_batch_size=query_batch_size,
                vocab_batch_size=sparse_vocab_batch_size,
                vocab_cuis=cache["vocab_cuis"],
                deduplicate_by_cui=deduplicate_by_cui,
                cui_overfetch_factor=cui_overfetch_factor,
                show_progress=True,
            )

            sparse_results_cache[entity_type] = {
                "sparse_index": sparse_index,
                "sparse_scores": sparse_scores,
                "sparse_indices": sparse_indices,
            }

    logger.info("Prepared sparse cache for %d entity types", len(sparse_results_cache))
    return sparse_results_cache


def select_best_result(results_df: pd.DataFrame, optimize_metric: str) -> pd.Series:
    """Select the best tuning result by metric values."""
    sort_columns = [optimize_metric]
    for metric in ("Acc@1", "Acc@5", "Acc@10", "Acc@20", "MRR"):
        if metric != optimize_metric and metric in results_df.columns:
            sort_columns.append(metric)
    return results_df.sort_values(sort_columns, ascending=[False] * len(sort_columns)).iloc[0]


def is_valid_sparse_param_combination(sparse_params: Dict) -> bool:
    """Validate sparse retrieval hyperparameters."""
    if "min_ngram" in sparse_params and "max_ngram" in sparse_params:
        return int(sparse_params["min_ngram"]) <= int(sparse_params["max_ngram"])
    return True


def iter_sparse_param_grid(param_grid: Dict[str, Iterable[float]]):
    """Iterate over sparse retrieval hyperparameter combinations."""
    sparse_param_names = [name for name in param_grid.keys() if name != "dense_weight"]
    dense_weight_values = list(param_grid["dense_weight"])
    sparse_value_lists = [list(param_grid[name]) for name in sparse_param_names]

    rows = []
    for values in product(*sparse_value_lists):
        sparse_params = dict(zip(sparse_param_names, values))
        if not is_valid_sparse_param_combination(sparse_params):
            logger.info("Skipping invalid sparse parameter combination: %s", sparse_params)
            continue
        for dense_weight in dense_weight_values:
            rows.append({**sparse_params, "dense_weight": dense_weight})
    return rows


def grid_search_sparse_params(
    entities_df,
    vocab_df,
    st_model,
    sparse_type: str,
    sparse_param_grid: Dict[str, Iterable[float]],
    sparse_fixed_index_params: Dict,
    topk: int,
    candidate_pool_size: int,
    query_batch_size: int,
    dense_vocab_batch_size: int,
    sparse_query_batch_size: int,
    sparse_vocab_batch_size: int,
    st_encode_batch_size: int,
    deduplicate_by_cui: bool = True,
    cui_overfetch_factor: int = 4,
    optimize_metric: str = "Acc@1",
):
    """Run sparse and fusion grid search on development data."""
    total_combinations = len(iter_sparse_param_grid(sparse_param_grid))
    logger.info(
        "Starting hybrid sparse grid search: sparse_type=%s, optimize_metric=%s, total_combinations=%d",
        sparse_type,
        optimize_metric,
        total_combinations,
    )
    with log_timed(logger, "Hybrid sparse grid search"):
        dense_cache = prepare_dense_dev_cache(
            entities_df=entities_df,
            vocab_df=vocab_df,
            st_model=st_model,
            candidate_pool_size=candidate_pool_size,
            query_batch_size=query_batch_size,
            dense_vocab_batch_size=dense_vocab_batch_size,
            st_encode_batch_size=st_encode_batch_size,
            deduplicate_by_cui=deduplicate_by_cui,
            cui_overfetch_factor=cui_overfetch_factor,
        )

        tuning_rows = []
        best_predictions_df = None

        sparse_param_names = [name for name in sparse_param_grid.keys() if name != "dense_weight"]
        sparse_param_pairs = []
        seen_pairs = set()
        for row in iter_sparse_param_grid(sparse_param_grid):
            sparse_params = tuple((name, row[name]) for name in sparse_param_names)
            if sparse_params not in seen_pairs:
                seen_pairs.add(sparse_params)
                sparse_param_pairs.append(dict(sparse_params))

        sparse_iterator = tqdm(sparse_param_pairs, desc=f"{sparse_type} param pairs", unit="pair")
        for sparse_params in sparse_iterator:
            logger.info("Evaluating sparse parameters: sparse_type=%s, sparse_params=%s", sparse_type, sparse_params)
            sparse_results_cache = prepare_sparse_results_cache(
                dense_cache=dense_cache,
                sparse_type=sparse_type,
                sparse_index_params={**sparse_fixed_index_params, **sparse_params},
                candidate_pool_size=candidate_pool_size,
                query_batch_size=sparse_query_batch_size,
                sparse_vocab_batch_size=sparse_vocab_batch_size,
                deduplicate_by_cui=deduplicate_by_cui,
                cui_overfetch_factor=cui_overfetch_factor,
            )

            dense_weight_values = list(sparse_param_grid["dense_weight"])
            weight_iterator = tqdm(
                dense_weight_values,
                desc=f"Fusion weights ({sparse_params})",
                unit="weight",
                leave=False,
            )
            for dense_weight in weight_iterator:
                sparse_weight = 1.0 - dense_weight
                predictions_df = make_hybrid_predictions_from_dense_cache_with_sparse(
                    dense_cache=dense_cache,
                    topk=topk,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    st_model=st_model,
                    st_encode_batch_size=st_encode_batch_size,
                    sparse_results_cache=sparse_results_cache,
                )

                metrics = evaluate_dev_predictions(predictions_df=predictions_df, data_df=entities_df)
                tuning_row = {
                    **sparse_params,
                    "dense_weight": dense_weight,
                    "sparse_weight": sparse_weight,
                    **metrics,
                }
                tuning_rows.append(tuning_row)

                if best_predictions_df is None:
                    best_predictions_df = predictions_df
                else:
                    current_df = pd.DataFrame(tuning_rows)
                    best_row = select_best_result(current_df, optimize_metric)
                    if all(best_row[name] == tuning_row[name] for name in sparse_params.keys()) and float(best_row["dense_weight"]) == float(dense_weight):
                        best_predictions_df = predictions_df

    results_df = pd.DataFrame(tuning_rows)
    best_row = select_best_result(results_df, optimize_metric)
    best_params = {
        **{name: best_row[name] for name in sparse_param_names},
        "DENSE_WEIGHT": float(best_row["dense_weight"]),
        "SPARSE_WEIGHT": float(best_row["sparse_weight"]),
    }
    return {
        "dense_cache": dense_cache,
        "results_df": results_df,
        "best_params": best_params,
        "best_metrics": {
            metric: float(best_row[metric])
            for metric in ("Acc@1", "Acc@5", "Acc@10", "Acc@20", "MRR")
            if metric in best_row.index and pd.notna(best_row[metric])
        },
        "best_predictions_df": best_predictions_df,
    }
