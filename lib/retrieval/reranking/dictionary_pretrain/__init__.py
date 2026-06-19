"""Public API for dictionary-based cross-encoder pretraining."""

from .artifacts import (
    build_dictionary_pretrain_cache_metadata,
    load_dataframe_cache,
    save_dataframe_cache,
)
from .data import (
    assign_dictionary_pretrain_splits,
    build_dictionary_pretrain_concepts,
    build_dictionary_pretrain_queries,
    build_dictionary_pretrain_vocab_subset,
)
from .fingerprints import (
    fingerprint_candidate_text_map,
    fingerprint_dictionary_pretrain_dataframe,
)
from .model_io import load_cross_encoder_from_pretrained
from .pairwise import build_dictionary_pretrain_pairwise_data
from .retriever_cache import (
    build_dictionary_pretrain_retriever_cache,
    flatten_dictionary_pretrain_retriever_cache,
    infer_dictionary_pretrain_cache_topk,
    load_dictionary_pretrain_retriever_cache,
    save_dictionary_pretrain_retriever_cache,
    trim_dictionary_pretrain_retriever_cache,
)
from .training import (
    DictionaryPretrainEvalCallback,
    build_cross_encoder_pretrain_dataset,
    build_dictionary_pretrain_training_arguments,
    compute_dictionary_pretrain_ranking_metrics,
    train_dictionary_pretrain_cross_encoder,
)

__all__ = [
    "DictionaryPretrainEvalCallback",
    "assign_dictionary_pretrain_splits",
    "build_cross_encoder_pretrain_dataset",
    "build_dictionary_pretrain_cache_metadata",
    "build_dictionary_pretrain_concepts",
    "build_dictionary_pretrain_pairwise_data",
    "build_dictionary_pretrain_queries",
    "build_dictionary_pretrain_retriever_cache",
    "build_dictionary_pretrain_training_arguments",
    "build_dictionary_pretrain_vocab_subset",
    "compute_dictionary_pretrain_ranking_metrics",
    "fingerprint_candidate_text_map",
    "fingerprint_dictionary_pretrain_dataframe",
    "flatten_dictionary_pretrain_retriever_cache",
    "infer_dictionary_pretrain_cache_topk",
    "load_cross_encoder_from_pretrained",
    "load_dataframe_cache",
    "load_dictionary_pretrain_retriever_cache",
    "save_dataframe_cache",
    "save_dictionary_pretrain_retriever_cache",
    "train_dictionary_pretrain_cross_encoder",
    "trim_dictionary_pretrain_retriever_cache",
]
