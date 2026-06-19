"""Retrieval metric computation helpers."""

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

DOCUMENT_ID_COL = "document_id"
SPANS_COL = "spans"
RANK_COL = "rank"
SAMPLE_PK_COL = "pk"
TRUE_CUI_COL = "UMLS_CUI"
PREDICTION_COL = "prediction"


def create_row_primary_key(row):
    """Create a stable row key from document and span fields."""
    return f"{row[DOCUMENT_ID_COL]}|{row[SPANS_COL]}"


def create_sample_pk2_true_cui_map(df: pd.DataFrame, true_cui_column: str) -> Dict[str, str]:
    """Map sample primary keys to gold CUIs."""
    return dict(zip(df[SAMPLE_PK_COL], df[true_cui_column]))


def calculate_metrics(pred_df: pd.DataFrame, sample_pk2true_cui: Dict[str, str]):
    """Compute Acc@1, Acc@5, and MRR from prediction rows."""
    logger.info(
        "Calculating ranking metrics: num_prediction_rows=%d, num_samples=%d",
        len(pred_df),
        len(sample_pk2true_cui),
    )
    sample_id2min_true_predicted_rank = {}

    for _, row in pred_df.iterrows():
        sample_pk = row[SAMPLE_PK_COL]
        rank = row[RANK_COL]
        pred_cui = row[PREDICTION_COL]
        true_cui = sample_pk2true_cui[sample_pk]

        if pred_cui == true_cui:
            if sample_id2min_true_predicted_rank.get(sample_pk) is None:
                sample_id2min_true_predicted_rank[sample_pk] = rank
            sample_id2min_true_predicted_rank[sample_pk] = min(sample_id2min_true_predicted_rank[sample_pk], rank)

    max_rank = int(pred_df[RANK_COL].max()) if len(pred_df) > 0 else 0
    requested_cutoffs = [k for k in (1, 5, 10, 20) if k <= max_rank]
    acc_sums = {k: 0.0 for k in requested_cutoffs}
    mrr_sum = 0.0

    for sample_id in sample_pk2true_cui.keys():
        rank = sample_id2min_true_predicted_rank.get(sample_id, -1)
        if rank == -1:
            sample_acc = {k: 0.0 for k in requested_cutoffs}
            sample_mrr = 0.0
        else:
            sample_acc = {k: 1.0 if rank <= k else 0.0 for k in requested_cutoffs}
            sample_mrr = 1.0 / rank

        for cutoff in requested_cutoffs:
            acc_sums[cutoff] += sample_acc[cutoff]
        mrr_sum += sample_mrr

    num_samples = len(sample_pk2true_cui)
    metrics = {f"Acc@{cutoff}": acc_sums[cutoff] / num_samples for cutoff in requested_cutoffs}
    metrics["MRR"] = mrr_sum / num_samples
    logger.info("Calculated metrics: %s", metrics)
    return metrics
