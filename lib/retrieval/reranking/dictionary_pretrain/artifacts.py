"""Save, load, and fingerprint cached dictionary pretraining artifacts."""

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .fingerprints import _fingerprint_dataframe


def save_dataframe_cache(
    df: pd.DataFrame,
    metadata: Dict,
    output_dir,
    *,
    stem: str,
) -> Dict[str, str]:
    """Save a cached dataframe with preview and metadata files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / f"{stem}.parquet"
    preview_path = output_dir / f"{stem}_preview.tsv"
    metadata_path = output_dir / f"{stem}_metadata.json"

    serializable_df = df.copy()
    for column in ("pseudo_queries",):
        if column in serializable_df.columns:
            serializable_df[column] = serializable_df[column].map(
                lambda values: " | ".join(values) if isinstance(values, list) else str(values)
            )

    serializable_df.to_parquet(cache_path, index=False)
    serializable_df.head(200).to_csv(preview_path, sep="\t", index=False)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "cache_path": str(cache_path),
        "preview_path": str(preview_path),
        "metadata_path": str(metadata_path),
    }


def load_dataframe_cache(cache_path, metadata_path: str | None = None) -> tuple[pd.DataFrame, Dict]:
    """Load a cached dataframe and its optional metadata."""
    cache_df = pd.read_parquet(cache_path)
    if "pseudo_queries" in cache_df.columns:
        cache_df["pseudo_queries"] = cache_df["pseudo_queries"].fillna("").map(
            lambda value: [part.strip() for part in str(value).split("|") if part.strip()]
        )
    metadata = {}
    if metadata_path is not None and Path(metadata_path).exists():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return cache_df, metadata


def build_dictionary_pretrain_cache_metadata(
    *,
    vocab_df: pd.DataFrame,
    cfg: Dict,
    extra: Dict | None = None,
) -> Dict:
    """Build metadata used to validate dictionary pretraining caches."""
    metadata = {
        "vocab_fingerprint": _fingerprint_dataframe(
            vocab_df,
            columns=["concept_name", "CUI", "semantic_type", "lang"],
        ),
        "config": json.loads(json.dumps(cfg)),
    }
    if extra:
        metadata.update(extra)
    return metadata
