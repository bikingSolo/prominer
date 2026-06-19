"""Vocabulary filtering and enrichment utilities."""

import hashlib
import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

ENRICHED_VOCAB_CACHE: Dict[tuple, pd.DataFrame] = {}


def _normalize_lang_code(value: str | None) -> str:
    lang = str(value or "").strip().upper()
    if lang == "EN":
        return "ENG"
    if lang == "RU":
        return "RUS"
    return lang


def _fingerprint_dataframe(df: pd.DataFrame, *, columns: list[str] | None = None) -> str:
    fingerprint_df = df if columns is None else df[[column for column in columns if column in df.columns]]
    hashed = pd.util.hash_pandas_object(fingerprint_df, index=True).values.tobytes()
    digest = hashlib.sha256()
    digest.update("|".join(fingerprint_df.columns.tolist()).encode("utf-8"))
    digest.update(str(fingerprint_df.shape).encode("utf-8"))
    digest.update(hashed)
    return digest.hexdigest()


def filter_vocab_for_dataset_language(
    vocab_df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Filter vocabulary rows for a dataset language."""
    dataset_name = str(dataset_name).lower()
    if dataset_name != "en":
        return vocab_df

    if "lang" not in vocab_df.columns:
        raise ValueError("vocab_df must contain a 'lang' column for dataset-specific language filtering.")

    normalized_lang = vocab_df["lang"].map(_normalize_lang_code)
    filtered_vocab_df = vocab_df[normalized_lang == "ENG"].copy()
    logger.info(
        "Filtered vocabulary for dataset=%s to English rows only: before_shape=%s, after_shape=%s",
        dataset_name,
        vocab_df.shape,
        filtered_vocab_df.shape,
    )
    return filtered_vocab_df.reset_index(drop=True)


def build_vocab_enrichment_rows(
    vocab_df: pd.DataFrame,
    entities_df: pd.DataFrame,
    enrichment_mode: str,
    *,
    text_column: str = "text",
    lang_value: str | None = None,
) -> pd.DataFrame:
    """Build extra vocabulary rows from train/dev entity mentions."""
    if enrichment_mode not in {"all_unique_pairs", "missing_cui_only"}:
        raise ValueError("ENRICHMENT_MODE must be 'all_unique_pairs' or 'missing_cui_only'.")

    logger.info("Building vocabulary enrichment rows with mode=%s", enrichment_mode)

    required_columns = {text_column, "UMLS_CUI", "entity_type"}
    missing_columns = required_columns.difference(entities_df.columns)
    if missing_columns:
        raise ValueError(f"entities_df is missing required columns for enrichment: {sorted(missing_columns)}")

    mention_df = entities_df[[text_column, "UMLS_CUI", "entity_type"]].dropna().copy()
    mention_df["CUI"] = mention_df["UMLS_CUI"].astype(str)
    mention_df = mention_df[mention_df["CUI"] != "CUILESS"].copy()
    mention_df["concept_name"] = mention_df[text_column].astype(str)
    mention_df["semantic_type"] = mention_df["entity_type"].astype(str)
    mention_df = mention_df[mention_df["concept_name"].astype(str).str.len() > 0].copy()
    mention_df = mention_df[["concept_name", "CUI", "semantic_type"]].drop_duplicates()

    if enrichment_mode == "missing_cui_only":
        existing_cuis = set(vocab_df["CUI"].dropna().astype(str).tolist())
        mention_df = mention_df[~mention_df["CUI"].isin(existing_cuis)].copy()

    existing_pair_keys = set(
        zip(
            vocab_df["concept_name"].fillna("").astype(str),
            vocab_df["CUI"].fillna("").astype(str),
            vocab_df["semantic_type"].fillna("").astype(str),
        )
    )
    mention_df["pair_key"] = list(zip(mention_df["concept_name"], mention_df["CUI"], mention_df["semantic_type"]))
    mention_df = mention_df[~mention_df["pair_key"].isin(existing_pair_keys)].drop(columns=["pair_key"]).reset_index(drop=True)

    enrichment_df = pd.DataFrame(index=mention_df.index, columns=vocab_df.columns)
    for column in enrichment_df.columns:
        if column in mention_df.columns:
            enrichment_df[column] = mention_df[column].values
    if "lang" in enrichment_df.columns and lang_value is not None:
        enrichment_df["lang"] = str(lang_value)

    logger.info("Prepared %d new vocabulary rows", len(enrichment_df))
    return enrichment_df.reset_index(drop=True)


def enrich_vocab_with_oov_train_dev_terms(
    base_vocab_df: pd.DataFrame,
    train_dev_entities_df: pd.DataFrame,
    *,
    text_column: str = "text",
    lang_value: str | None = None,
) -> pd.DataFrame:
    """Add train/dev out-of-vocabulary mentions to the vocabulary."""
    logger.info("Applying default OOV train/dev vocabulary enrichment")
    enrichment_df = build_vocab_enrichment_rows(
        vocab_df=base_vocab_df,
        entities_df=train_dev_entities_df,
        enrichment_mode="missing_cui_only",
        text_column=text_column,
        lang_value=lang_value,
    )
    enriched_vocab_df = pd.concat([base_vocab_df, enrichment_df], ignore_index=True)
    logger.info(
        "Default OOV train/dev enrichment finished: added_rows=%d, final_shape=%s",
        len(enrichment_df),
        enriched_vocab_df.shape,
    )
    return enriched_vocab_df


def prepare_experiment_vocab(
    base_vocab_df: pd.DataFrame,
    enrichment_entities_df: pd.DataFrame,
    cfg: Dict,
    *,
    text_column: str = "text",
    lang_value: str | None = None,
) -> pd.DataFrame:
    """Apply optional all-pair train/dev vocabulary enrichment."""
    if not bool(cfg.get("ENRICH_VOCABULARY", False)):
        return base_vocab_df

    enrichment_mode = "all_unique_pairs"
    cache_key = (
        enrichment_mode,
        text_column,
        _fingerprint_dataframe(base_vocab_df, columns=["concept_name", "CUI", "semantic_type", "lang"]),
        _fingerprint_dataframe(enrichment_entities_df, columns=[text_column, "UMLS_CUI", "entity_type"]),
        None if lang_value is None else str(lang_value),
    )
    if cache_key in ENRICHED_VOCAB_CACHE:
        logger.info("Using cached enriched vocabulary for mode=%s", enrichment_mode)
        return ENRICHED_VOCAB_CACHE[cache_key]

    enrichment_df = build_vocab_enrichment_rows(
        vocab_df=base_vocab_df,
        entities_df=enrichment_entities_df,
        enrichment_mode=enrichment_mode,
        text_column=text_column,
        lang_value=lang_value,
    )
    enriched_vocab_df = pd.concat([base_vocab_df, enrichment_df], ignore_index=True)
    ENRICHED_VOCAB_CACHE[cache_key] = enriched_vocab_df
    logger.info(
        "Vocabulary enrichment finished: mode=%s, added_rows=%d, final_shape=%s",
        enrichment_mode,
        len(enrichment_df),
        enriched_vocab_df.shape,
    )
    return enriched_vocab_df
