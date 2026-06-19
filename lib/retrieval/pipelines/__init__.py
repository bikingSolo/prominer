"""Reusable retrieval prediction pipelines."""

from .dense_only import (
    evaluate_dense_retrieval,
    get_dense_type_resource,
    make_dense_predictions,
    predict_dense_to_path,
)
from .late_fusion import (
    get_type_resource_with_sparse,
    make_hybrid_predictions_with_sparse,
)
