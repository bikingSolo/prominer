"""Shared protocol and factory for sparse indexes."""

from typing import Dict

import numpy as np

from .bm25 import BM25Index
from .char_tfidf import CharTfidfIndex

SPARSE_INDEX_TYPES = {
    "bm25": BM25Index,
    "char_tfidf": CharTfidfIndex,
}


def build_sparse_index(sparse_type: str, documents: np.ndarray, sparse_index_params: Dict):
    """Build a sparse retrieval index from a method name."""
    sparse_type = str(sparse_type)
    index_cls = SPARSE_INDEX_TYPES.get(sparse_type)
    if index_cls is None:
        raise ValueError(f"Unsupported sparse_type: {sparse_type}")
    return index_cls(documents=documents, **sparse_index_params)
