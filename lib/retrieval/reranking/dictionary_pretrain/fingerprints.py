"""Compute stable fingerprints for dictionary pretraining inputs."""

import hashlib
import json
from typing import Dict

import pandas as pd


def fingerprint_dictionary_pretrain_dataframe(df: pd.DataFrame, *, columns: list[str]) -> str:
    """Fingerprint selected dataframe columns for cache invalidation."""
    return _fingerprint_dataframe(df, columns=columns)


def fingerprint_candidate_text_map(candidate_text_map: Dict[tuple[str, str], str]) -> str:
    """Fingerprint candidate-context text overrides."""
    digest = hashlib.sha256()
    for (cui, semantic_type), candidate_text in sorted(candidate_text_map.items()):
        digest.update(str(cui).encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(semantic_type).encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(candidate_text).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fingerprint_dataframe(df: pd.DataFrame, *, columns: list[str]) -> str:
    """Hash selected dataframe content in a stable way."""
    available_columns = [column for column in columns if column in df.columns]
    fingerprint_df = df[available_columns].copy()
    for column in available_columns:
        if fingerprint_df[column].dtype == "object":
            fingerprint_df[column] = fingerprint_df[column].map(
                lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (list, dict, tuple, set))
                else value
            )
    hashed = pd.util.hash_pandas_object(fingerprint_df, index=True).values.tobytes()
    digest = hashlib.sha256()
    digest.update("|".join(available_columns).encode("utf-8"))
    digest.update(str(fingerprint_df.shape).encode("utf-8"))
    digest.update(hashed)
    return digest.hexdigest()
