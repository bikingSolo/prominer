"""Persistence helpers for candidate-context caches."""

import json
from pathlib import Path
from typing import Dict

import pandas as pd


def build_candidate_text_map(candidate_context_df: pd.DataFrame) -> Dict[tuple[str, str], str]:
    """Map candidate keys to prepared candidate-context text."""
    required_columns = {"CUI", "semantic_type", "candidate_text"}
    missing_columns = required_columns.difference(candidate_context_df.columns)
    if missing_columns:
        raise ValueError(f"candidate_context_df is missing required columns: {sorted(missing_columns)}")
    cuis = candidate_context_df["CUI"].astype(str).tolist()
    semantic_types = candidate_context_df["semantic_type"].astype(str).tolist()
    candidate_texts = candidate_context_df["candidate_text"].astype(str).tolist()
    return {
        (cui, semantic_type): candidate_text
        for cui, semantic_type, candidate_text in zip(cuis, semantic_types, candidate_texts)
    }


def save_candidate_context_cache(
    candidate_context_df: pd.DataFrame,
    metadata: Dict[str, float],
    output_dir,
    *,
    stem: str = "candidate_context",
) -> Dict[str, str]:
    """Save a candidate-context cache with metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / f"{stem}.parquet"
    preview_path = output_dir / f"{stem}_preview.parquet"
    metadata_path = output_dir / f"{stem}_metadata.json"

    serializable_df = candidate_context_df.copy()
    if "selected_aliases" in serializable_df.columns:
        serializable_df["selected_aliases"] = serializable_df["selected_aliases"].map(
            lambda values: " | ".join(values) if isinstance(values, list) else str(values)
        )
    if "selected_aliases_normalized" in serializable_df.columns:
        serializable_df["selected_aliases_normalized"] = serializable_df["selected_aliases_normalized"].map(
            lambda values: " | ".join(values) if isinstance(values, list) else str(values)
        )
    if "languages" in serializable_df.columns:
        serializable_df["languages"] = serializable_df["languages"].map(
            lambda values: ",".join(values) if isinstance(values, list) else str(values)
        )

    serializable_df.to_parquet(cache_path, index=False)
    serializable_df.head(200).to_parquet(preview_path, index=False)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "cache_path": str(cache_path),
        "preview_path": str(preview_path),
        "metadata_path": str(metadata_path),
    }


def load_candidate_context_cache(cache_path, metadata_path: str | None = None) -> tuple[pd.DataFrame, Dict]:
    """Load a candidate-context cache with metadata."""
    cache_path = Path(cache_path)
    if cache_path.suffix == ".parquet":
        candidate_context_df = pd.read_parquet(cache_path)
    else:
        candidate_context_df = pd.read_csv(cache_path, sep="\t")

    if "selected_aliases" in candidate_context_df.columns:
        candidate_context_df["selected_aliases"] = candidate_context_df["selected_aliases"].fillna("").map(
            lambda value: [part.strip() for part in str(value).split("|") if part.strip()]
        )
    if "selected_aliases_normalized" in candidate_context_df.columns:
        candidate_context_df["selected_aliases_normalized"] = candidate_context_df[
            "selected_aliases_normalized"
        ].fillna("").map(
            lambda value: [part.strip() for part in str(value).split("|") if part.strip()]
        )
    if "languages" in candidate_context_df.columns:
        candidate_context_df["languages"] = candidate_context_df["languages"].fillna("").map(
            lambda value: [part.strip() for part in str(value).split(",") if part.strip()]
        )

    metadata = {}
    if metadata_path is not None and Path(metadata_path).exists():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return candidate_context_df, metadata
