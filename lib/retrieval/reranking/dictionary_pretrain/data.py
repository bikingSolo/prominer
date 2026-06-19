"""Build dictionary pretraining concepts, pseudo-queries, and train/dev splits."""

import hashlib
import logging
import random
from typing import Dict

import pandas as pd

from lib.data.text_preprocessing import preprocess_texts

from .constants import PRETRAIN_QUERY_COLUMNS


logger = logging.getLogger(__name__)


def _normalize_lang_code(value: str | None) -> str:
    """Normalize short language codes to vocabulary language tags."""
    language = str(value or "").strip().upper()
    if language == "RU":
        language = "RUS"
    elif language == "EN":
        language = "ENG"
    return language


def _hash_to_unit_interval(text: str) -> float:
    """Map text deterministically to a unit-interval value."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16 ** 16 - 1)


def _prepare_pretrain_vocab_records(vocab_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize raw vocabulary rows for pretraining."""
    required_columns = {"CUI", "semantic_type", "concept_name"}
    missing_columns = required_columns.difference(vocab_df.columns)
    if missing_columns:
        raise ValueError(f"vocab_df is missing required columns: {sorted(missing_columns)}")

    records_df = vocab_df[["CUI", "semantic_type", "concept_name"]].copy()
    records_df["CUI"] = records_df["CUI"].astype(str)
    records_df["semantic_type"] = records_df["semantic_type"].astype(str)
    records_df["concept_name_raw"] = records_df["concept_name"].fillna("").astype(str).str.strip()
    records_df["normalized_concept_name"] = preprocess_texts(records_df["concept_name_raw"].tolist())
    if "lang" in vocab_df.columns:
        records_df["normalized_lang"] = vocab_df["lang"].map(_normalize_lang_code)
    else:
        records_df["normalized_lang"] = ""
    return records_df


def _build_representative_name_table(prepared_vocab_df: pd.DataFrame) -> pd.DataFrame:
    """Select one stable representative name per concept."""
    concept_keys_df = prepared_vocab_df[["CUI", "semantic_type"]].drop_duplicates().copy()

    representative_df = prepared_vocab_df.loc[
        prepared_vocab_df["concept_name_raw"].astype(str).str.len() > 0,
        ["CUI", "semantic_type", "concept_name_raw"],
    ].copy()
    if representative_df.empty:
        concept_keys_df["representative_name"] = ""
        return concept_keys_df

    representative_df["concept_name_len"] = representative_df["concept_name_raw"].str.len()
    representative_df["concept_name_casefold"] = representative_df["concept_name_raw"].str.casefold()
    representative_df = representative_df.sort_values(
        ["semantic_type", "CUI", "concept_name_len", "concept_name_casefold", "concept_name_raw"],
        kind="stable",
    )
    representative_df = representative_df.drop_duplicates(["CUI", "semantic_type"], keep="first")
    representative_df = representative_df.rename(columns={"concept_name_raw": "representative_name"})
    representative_df = representative_df[["CUI", "semantic_type", "representative_name"]]

    result_df = concept_keys_df.merge(representative_df, on=["CUI", "semantic_type"], how="left")
    result_df["representative_name"] = result_df["representative_name"].fillna("")
    return result_df


def _build_candidate_doc_table_from_prepared(
    prepared_vocab_df: pd.DataFrame,
    *,
    candidate_text_map: Dict[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    """Build candidate texts from prepared vocabulary records."""
    candidate_text_map = {} if candidate_text_map is None else candidate_text_map

    result_df = _build_representative_name_table(prepared_vocab_df)
    result_df["candidate_text"] = preprocess_texts(result_df["representative_name"].tolist())
    if candidate_text_map:
        candidate_text_map_df = pd.DataFrame.from_records(
            (
                (str(cui), str(semantic_type), str(candidate_text))
                for (cui, semantic_type), candidate_text in candidate_text_map.items()
            ),
            columns=["CUI", "semantic_type", "candidate_text_override"],
        )
        result_df = result_df.merge(candidate_text_map_df, on=["CUI", "semantic_type"], how="left")
        override_mask = result_df["candidate_text_override"].notna()
        result_df.loc[override_mask, "candidate_text"] = (
            result_df.loc[override_mask, "candidate_text_override"].astype(str).str.strip()
        )
        result_df = result_df.drop(columns=["candidate_text_override"])

    result_df["candidate_text"] = result_df["candidate_text"].astype(str).str.strip()
    result_df = result_df[result_df["candidate_text"].astype(str).str.len() > 0].copy()
    result_df = result_df.sort_values(["semantic_type", "CUI"], kind="stable").reset_index(drop=True)
    return result_df


def _build_concept_synonym_pool(prepared_vocab_df: pd.DataFrame) -> pd.DataFrame:
    """Collect normalized synonyms for each concept."""
    synonym_df = prepared_vocab_df.loc[
        prepared_vocab_df["normalized_concept_name"].astype(str).str.len() > 0,
        ["CUI", "semantic_type", "normalized_concept_name", "normalized_lang"],
    ].copy()
    if synonym_df.empty:
        return pd.DataFrame(
            columns=["CUI", "semantic_type", "synonym_texts", "synonym_langs", "num_vocab_synonyms"]
        )

    synonym_df = synonym_df.drop_duplicates(
        subset=["CUI", "semantic_type", "normalized_concept_name"],
        keep="first",
    ).copy()
    synonym_df["synonym_len"] = synonym_df["normalized_concept_name"].str.len()
    synonym_df["synonym_casefold"] = synonym_df["normalized_concept_name"].str.casefold()
    synonym_df = synonym_df.sort_values(
        ["semantic_type", "CUI", "synonym_len", "synonym_casefold", "normalized_concept_name"],
        kind="stable",
    )

    grouped_df = (
        synonym_df.groupby(["CUI", "semantic_type"], sort=False, dropna=False)
        .agg(
            synonym_texts=("normalized_concept_name", list),
            synonym_langs=("normalized_lang", list),
            num_vocab_synonyms=("normalized_concept_name", "size"),
        )
        .reset_index()
    )
    return grouped_df


def _select_pseudo_queries(
    synonym_texts: list[str],
    synonym_langs: list[str],
    *,
    preferred_query_languages: list[str],
    max_pseudo_queries_per_cui: int,
    min_pseudo_queries_per_cui: int,
) -> list[str]:
    """Select stable synonym texts to use as pseudo-queries."""
    if not synonym_texts:
        return []

    limit = min(max(int(max_pseudo_queries_per_cui), 1), len(synonym_texts))
    selected_synonyms: list[str] = []
    selected_indices: set[int] = set()

    for preferred_language in preferred_query_languages:
        matching_indices = [
            index
            for index, language in enumerate(synonym_langs)
            if language == preferred_language
        ]
        if not matching_indices:
            continue
        best_index = matching_indices[0]
        selected_synonyms.append(synonym_texts[best_index])
        selected_indices.add(best_index)
        break

    for index, synonym_text in enumerate(synonym_texts):
        if len(selected_synonyms) >= limit:
            break
        if index in selected_indices:
            continue
        selected_synonyms.append(synonym_text)

    if len(selected_synonyms) < max(int(min_pseudo_queries_per_cui), 1):
        return []
    return selected_synonyms


def build_dictionary_pretrain_vocab_subset(
    vocab_df: pd.DataFrame,
    train_dev_entities_df: pd.DataFrame,
    *,
    num_extra_cuis: int = 0,
    seed: int = 42,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Select the vocabulary subset used for dictionary pretraining."""
    required_vocab_columns = {"CUI", "concept_name"}
    missing_vocab_columns = required_vocab_columns.difference(vocab_df.columns)
    if missing_vocab_columns:
        raise ValueError(f"vocab_df is missing required columns: {sorted(missing_vocab_columns)}")

    required_entity_columns = {"UMLS_CUI"}
    missing_entity_columns = required_entity_columns.difference(train_dev_entities_df.columns)
    if missing_entity_columns:
        raise ValueError(f"train_dev_entities_df is missing required columns: {sorted(missing_entity_columns)}")

    vocab_df = vocab_df.copy()
    vocab_df["CUI"] = vocab_df["CUI"].astype(str)
    train_dev_cuis = {
        str(cui)
        for cui in train_dev_entities_df["UMLS_CUI"].dropna().astype(str).tolist()
        if str(cui) != "CUILESS"
    }

    russian_cuis = set()
    if "lang" in vocab_df.columns:
        normalized_lang = vocab_df["lang"].map(_normalize_lang_code)
        russian_cuis = set(vocab_df.loc[normalized_lang == "RUS", "CUI"].astype(str).tolist())

    selected_cuis = set(train_dev_cuis) | set(russian_cuis)
    all_cuis = set(vocab_df["CUI"].dropna().astype(str).tolist())
    remaining_cuis = sorted(all_cuis.difference(selected_cuis))

    extra_cui_count = max(int(num_extra_cuis), 0)
    if extra_cui_count > 0 and remaining_cuis:
        rng = random.Random(int(seed))
        sampled_extra_cuis = rng.sample(remaining_cuis, k=min(extra_cui_count, len(remaining_cuis)))
    else:
        sampled_extra_cuis = []

    selected_cuis.update(sampled_extra_cuis)
    subset_df = vocab_df[vocab_df["CUI"].isin(selected_cuis)].copy().reset_index(drop=True)
    stats = {
        "num_vocab_rows": int(len(subset_df)),
        "num_selected_cuis": int(len(selected_cuis)),
        "num_train_dev_cuis": int(len(train_dev_cuis)),
        "num_cuis_with_russian_synonyms": int(len(russian_cuis)),
        "num_extra_cuis": int(len(sampled_extra_cuis)),
    }
    logger.info("Prepared dictionary pretrain vocab subset: %s", stats)
    return subset_df, stats


def build_dictionary_pretrain_concepts(
    vocab_df: pd.DataFrame,
    *,
    candidate_text_map: Dict[tuple[str, str], str] | None = None,
    max_pseudo_queries_per_cui: int = 5,
    min_pseudo_queries_per_cui: int = 1,
    preferred_query_languages: list[str] | None = None,
) -> pd.DataFrame:
    """Create concept-level rows with candidate texts and pseudo-query synonyms."""
    prepared_vocab_df = _prepare_pretrain_vocab_records(vocab_df)
    candidate_doc_df = _build_candidate_doc_table_from_prepared(
        prepared_vocab_df,
        candidate_text_map=candidate_text_map,
    )
    preferred_query_languages = [_normalize_lang_code(language) for language in (preferred_query_languages or [])]
    synonym_pool_df = _build_concept_synonym_pool(prepared_vocab_df)
    concepts_base_df = candidate_doc_df.merge(
        synonym_pool_df,
        on=["CUI", "semantic_type"],
        how="left",
    )
    concepts_base_df = concepts_base_df[
        concepts_base_df["synonym_texts"].map(lambda value: isinstance(value, list) and len(value) > 0)
    ].copy()

    concepts_base_df["pseudo_queries"] = [
        _select_pseudo_queries(
            synonym_texts,
            synonym_langs,
            preferred_query_languages=preferred_query_languages,
            max_pseudo_queries_per_cui=max_pseudo_queries_per_cui,
            min_pseudo_queries_per_cui=min_pseudo_queries_per_cui,
        )
        for synonym_texts, synonym_langs in zip(
            concepts_base_df["synonym_texts"],
            concepts_base_df["synonym_langs"],
        )
    ]
    concepts_base_df = concepts_base_df[
        concepts_base_df["pseudo_queries"].map(lambda values: len(values) >= max(int(min_pseudo_queries_per_cui), 1))
    ].copy()
    concepts_base_df["num_pseudo_queries"] = concepts_base_df["pseudo_queries"].map(len).astype(int)

    result_df = concepts_base_df[
        [
            "CUI",
            "semantic_type",
            "candidate_text",
            "representative_name",
            "pseudo_queries",
            "num_pseudo_queries",
            "num_vocab_synonyms",
        ]
    ].sort_values(["semantic_type", "CUI"], kind="stable").reset_index(drop=True)
    logger.info("Prepared dictionary pretrain concepts: num_rows=%d", len(result_df))
    return result_df


def assign_dictionary_pretrain_splits(
    concepts_df: pd.DataFrame,
    *,
    validation_fraction: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign stable concept-level train/dev splits."""
    if concepts_df.empty:
        result_df = concepts_df.copy()
        result_df["split"] = pd.Series(dtype="object")
        return result_df

    validation_fraction = min(max(float(validation_fraction), 0.0), 0.5)
    result_df = concepts_df.copy()

    if validation_fraction <= 0.0:
        result_df["split"] = "train"
        return result_df

    split_values = []
    for cui, semantic_type in zip(result_df["CUI"], result_df["semantic_type"]):
        split_key = f"{seed}|{semantic_type}|{cui}"
        split_values.append("dev" if _hash_to_unit_interval(split_key) < validation_fraction else "train")
    result_df["split"] = split_values

    if result_df["split"].eq("dev").sum() == 0 and len(result_df) > 1:
        result_df.at[len(result_df) - 1, "split"] = "dev"
    if result_df["split"].eq("train").sum() == 0:
        result_df["split"] = "train"

    return result_df


def build_dictionary_pretrain_queries(concepts_df: pd.DataFrame) -> pd.DataFrame:
    """Expand concept pseudo-queries into query-level training rows."""
    required_columns = {"CUI", "semantic_type", "candidate_text", "pseudo_queries", "split"}
    missing_columns = required_columns.difference(concepts_df.columns)
    if missing_columns:
        raise ValueError(f"concepts_df is missing required columns: {sorted(missing_columns)}")

    if concepts_df.empty:
        return pd.DataFrame(columns=PRETRAIN_QUERY_COLUMNS)

    result_df = concepts_df[
        ["CUI", "semantic_type", "candidate_text", "pseudo_queries", "split"]
    ].copy()
    result_df = result_df.explode("pseudo_queries", ignore_index=True)
    result_df = result_df.rename(columns={"pseudo_queries": "query_text"})
    result_df["query_text"] = result_df["query_text"].astype(str)
    result_df["query_idx"] = (
        result_df.groupby(["semantic_type", "CUI"], sort=False).cumcount() + 1
    )
    result_df["query_id"] = (
        result_df["semantic_type"].astype(str)
        + "|"
        + result_df["CUI"].astype(str)
        + "|"
        + result_df["query_idx"].astype(str)
    )
    result_df = result_df[
        ["query_id", "query_text", "CUI", "semantic_type", "candidate_text", "split"]
    ].sort_values(["split", "semantic_type", "CUI", "query_id"], kind="stable").reset_index(drop=True)
    logger.info("Prepared dictionary pretrain queries: num_rows=%d", len(result_df))
    return result_df
