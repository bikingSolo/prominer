"""Shared dataframe schemas for dictionary pretraining artifacts."""

PRETRAIN_QUERY_COLUMNS = [
    "query_id",
    "query_text",
    "CUI",
    "semantic_type",
    "candidate_text",
    "split",
]

PRETRAIN_PAIR_COLUMNS = [
    "query",
    "candidate_text",
    "label",
    "query_id",
    "query_text",
    "gold_cui",
    "candidate_cui",
    "semantic_type",
    "split",
    "candidate_rank",
    "retriever_score",
]
