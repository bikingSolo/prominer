"""Candidate-context construction for cross-encoder inputs."""

import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Dict

import pandas as pd

from lib.data.text_preprocessing import preprocess_text

from lib.retrieval.reranking.candidate_aliases import (
    DEFAULT_GROUP_LIMITS,
    _classify_alias,
    _language_priority,
    _normalize_language,
    _pick_representative,
    _sanitize_alias_text,
    _select_diverse_aliases,
    estimate_alias_length_threshold,
    is_likely_abbreviation,
    normalize_alias_for_dedup,
)
from lib.retrieval.reranking.candidate_context_cache import (
    build_candidate_text_map,
    load_candidate_context_cache,
    save_candidate_context_cache,
)


logger = logging.getLogger(__name__)


def _prepare_alias_records(group_df: pd.DataFrame, allowed_languages: list[str]) -> tuple[list[Dict], list[str], int]:
    """Normalize usable alias records for one concept group."""
    alias_records = []
    raw_aliases = []
    for alias_row in group_df.itertuples(index=False):
        raw_alias_text = str(alias_row.concept_name).strip()
        if not raw_alias_text:
            continue
        raw_aliases.append(raw_alias_text)
        sanitized_alias_text = _sanitize_alias_text(raw_alias_text)
        if not sanitized_alias_text:
            continue
        normalized_alias_text = preprocess_text(sanitized_alias_text)
        if not normalized_alias_text:
            continue
        alias_records.append(
            {
                "raw_text": sanitized_alias_text,
                "normalized_text": normalized_alias_text,
                "lang": _normalize_language(getattr(alias_row, "lang", None)),
            }
        )

    if allowed_languages:
        alias_records = [record for record in alias_records if record["lang"] in allowed_languages]
    removed_noisy = max(0, len(raw_aliases) - len(alias_records))
    return alias_records, raw_aliases, removed_noisy


def _deduplicate_alias_records(
    alias_records: list[Dict],
    *,
    alias_length_threshold: int,
    preferred_languages: list[str],
) -> tuple[list[str], dict[str, str], dict[str, str], dict[str, str], int, int]:
    """Deduplicate aliases and keep one fallback alias if length filtering removes all."""
    deduplicated_aliases = []
    alias2language = {}
    alias2normalized = {}
    alias2dedup_norm = {}
    seen_norms = set()
    removed_duplicates = 0
    removed_long = 0

    for record in alias_records:
        alias = record["raw_text"]
        norm = normalize_alias_for_dedup(alias)
        if not norm:
            continue
        if norm in seen_norms:
            removed_duplicates += 1
            continue
        seen_norms.add(norm)
        if len(alias) > alias_length_threshold:
            removed_long += 1
            continue
        deduplicated_aliases.append(alias)
        alias2language[alias] = record["lang"]
        alias2normalized[alias] = record["normalized_text"]
        alias2dedup_norm[alias] = norm

    if not deduplicated_aliases and alias_records:
        fallback_candidates = []
        seen_fallback_norms = set()
        for record in alias_records:
            alias = record["raw_text"]
            fallback_norm = normalize_alias_for_dedup(alias)
            if not fallback_norm or fallback_norm in seen_fallback_norms:
                continue
            seen_fallback_norms.add(fallback_norm)
            fallback_candidates.append(alias)
            alias2language.setdefault(alias, record["lang"])
            alias2normalized.setdefault(alias, record["normalized_text"])
            alias2dedup_norm.setdefault(alias, fallback_norm)

        fallback_aliases = sorted(
            fallback_candidates,
            key=lambda alias: (
                _language_priority(alias2language.get(alias), preferred_languages),
                len(alias),
                alias.casefold(),
            ),
        )
        if fallback_aliases:
            deduplicated_aliases = [fallback_aliases[0]]

    return deduplicated_aliases, alias2language, alias2normalized, alias2dedup_norm, removed_duplicates, removed_long


def _select_representative_alias(
    deduplicated_aliases: list[str],
    *,
    alias2language: dict[str, str],
    preferred_languages: list[str],
) -> str:
    """Select the main display alias for a concept."""
    alias2language_priority = {
        alias: _language_priority(alias2language.get(alias), preferred_languages)
        for alias in deduplicated_aliases
    }
    representative_candidates = sorted(
        deduplicated_aliases,
        key=lambda alias: (
            alias2language_priority[alias],
            is_likely_abbreviation(alias),
            len(alias),
            alias.casefold(),
        ),
    )
    best_language_priority = alias2language_priority[representative_candidates[0]]
    top_language_candidates = [
        alias
        for alias in representative_candidates
        if alias2language_priority[alias] == best_language_priority
    ]
    return _pick_representative(top_language_candidates)


def _group_aliases_by_kind(deduplicated_aliases: list[str], *, representative_norm: str, alias2dedup_norm: dict[str, str]) -> dict[str, list[str]]:
    """Group non-representative aliases by coarse alias type."""
    grouped_aliases = {
        "abbreviations": [],
        "short_names": [],
        "multi_word": [],
        "long_variants": [],
    }
    for alias in deduplicated_aliases:
        if alias2dedup_norm.get(alias) == representative_norm:
            continue
        grouped_aliases[_classify_alias(alias)].append(alias)
    return grouped_aliases


def _select_aliases_for_missing_languages(
    *,
    deduplicated_aliases: list[str],
    representative: str,
    representative_norm: str,
    alias2language: dict[str, str],
    alias2dedup_norm: dict[str, str],
    preferred_languages: list[str],
    ordered_aliases: list[str],
    selected_groups: dict[str, list[str]],
    remaining_budget: int,
) -> int:
    """Reserve alias slots for languages not covered by the first selection pass."""
    covered_languages = {_normalize_language(alias2language.get(representative))}
    covered_languages.update(_normalize_language(alias2language.get(alias)) for alias in ordered_aliases)
    available_languages = []
    for alias in deduplicated_aliases:
        language = _normalize_language(alias2language.get(alias))
        if language not in available_languages:
            available_languages.append(language)

    missing_languages = [language for language in available_languages if language not in covered_languages and language != "UNK"]
    for missing_language in missing_languages:
        if remaining_budget <= 0:
            break
        ordered_alias_norms = {alias2dedup_norm.get(value, normalize_alias_for_dedup(value)) for value in ordered_aliases}
        language_candidates = [
            alias
            for alias in deduplicated_aliases
            if _normalize_language(alias2language.get(alias)) == missing_language
            and alias2dedup_norm.get(alias) != representative_norm
            and alias2dedup_norm.get(alias) not in ordered_alias_norms
        ]
        selected_language_aliases = _select_diverse_aliases(
            language_candidates,
            limit=1,
            representative=representative,
            preferred_languages=preferred_languages,
            alias_languages=alias2language,
        )
        if not selected_language_aliases:
            continue
        selected_alias = selected_language_aliases[0]
        ordered_aliases.append(selected_alias)
        remaining_budget -= 1
        target_group = _classify_alias(selected_alias)
        selected_groups.setdefault(target_group, [])
        selected_groups[target_group].append(selected_alias)
        covered_languages.add(missing_language)
    return remaining_budget


def _select_leftover_aliases(
    *,
    grouped_aliases: dict[str, list[str]],
    representative: str,
    alias2language: dict[str, str],
    alias2dedup_norm: dict[str, str],
    preferred_languages: list[str],
    ordered_aliases: list[str],
    remaining_budget: int,
) -> None:
    """Fill remaining alias slots from unselected aliases."""
    if remaining_budget <= 0:
        return

    leftovers = []
    already_selected = {alias2dedup_norm.get(alias, normalize_alias_for_dedup(alias)) for alias in ordered_aliases}
    for group_name in ("short_names", "multi_word", "long_variants", "abbreviations"):
        for alias in grouped_aliases[group_name]:
            alias_norm = alias2dedup_norm.get(alias, normalize_alias_for_dedup(alias))
            if alias_norm not in already_selected:
                leftovers.append(alias)
    ordered_aliases.extend(
        _select_diverse_aliases(
            leftovers,
            limit=remaining_budget,
            representative=representative,
            preferred_languages=preferred_languages,
            alias_languages=alias2language,
        )
    )


def _select_candidate_aliases(
    deduplicated_aliases: list[str],
    *,
    representative: str,
    representative_norm: str,
    grouped_aliases: dict[str, list[str]],
    alias2language: dict[str, str],
    alias2dedup_norm: dict[str, str],
    max_aliases: int,
    group_limits: Dict[str, int],
    preferred_languages: list[str],
) -> tuple[list[str], dict[str, list[str]]]:
    """Select ordered aliases for the final candidate text."""
    selected_groups = {}
    ordered_aliases = []
    remaining_budget = max(0, int(max_aliases))
    group_order = ("abbreviations", "short_names", "multi_word", "long_variants")

    for group_name in group_order:
        group_limit = min(int(group_limits.get(group_name, 0)), remaining_budget)
        selected = _select_diverse_aliases(
            grouped_aliases[group_name],
            limit=group_limit,
            representative=representative,
            preferred_languages=preferred_languages,
            alias_languages=alias2language,
        )
        selected_groups[group_name] = selected
        ordered_aliases.extend(selected)
        remaining_budget -= len(selected)

    remaining_budget = _select_aliases_for_missing_languages(
        deduplicated_aliases=deduplicated_aliases,
        representative=representative,
        representative_norm=representative_norm,
        alias2language=alias2language,
        alias2dedup_norm=alias2dedup_norm,
        preferred_languages=preferred_languages,
        ordered_aliases=ordered_aliases,
        selected_groups=selected_groups,
        remaining_budget=remaining_budget,
    )
    _select_leftover_aliases(
        grouped_aliases=grouped_aliases,
        representative=representative,
        alias2language=alias2language,
        alias2dedup_norm=alias2dedup_norm,
        preferred_languages=preferred_languages,
        ordered_aliases=ordered_aliases,
        remaining_budget=remaining_budget,
    )

    return ordered_aliases, selected_groups


def _build_candidate_text(
    *,
    normalized_representative: str,
    selected_groups: dict[str, list[str]],
    alias2normalized: dict[str, str],
) -> str:
    """Join representative and selected aliases into one candidate text."""
    alias_parts = []
    for group_name in ("abbreviations", "short_names", "multi_word", "long_variants"):
        group_aliases = [alias for alias in selected_groups.get(group_name, []) if alias]
        if group_aliases:
            normalized_group_aliases = [alias2normalized.get(alias, preprocess_text(alias)) for alias in group_aliases]
            alias_parts.append("; ".join(normalized_group_aliases))

    candidate_segments = [normalized_representative]
    if alias_parts:
        candidate_segments.extend(alias_parts)
    return "; ".join(segment for segment in candidate_segments if segment)


def _build_candidate_context_row(
    cui,
    semantic_type,
    group_df: pd.DataFrame,
    *,
    alias_length_threshold: int,
    max_aliases: int,
    group_limits: Dict[str, int],
    preferred_languages: list[str],
    allowed_languages: list[str],
) -> Dict | None:
    """Build one candidate-context cache row for a concept."""
    alias_records, raw_aliases, removed_noisy = _prepare_alias_records(group_df, allowed_languages)
    (
        deduplicated_aliases,
        alias2language,
        alias2normalized,
        alias2dedup_norm,
        removed_duplicates,
        removed_long,
    ) = _deduplicate_alias_records(
        alias_records,
        alias_length_threshold=alias_length_threshold,
        preferred_languages=preferred_languages,
    )
    if not deduplicated_aliases:
        return None

    representative = _select_representative_alias(
        deduplicated_aliases,
        alias2language=alias2language,
        preferred_languages=preferred_languages,
    )
    normalized_representative = alias2normalized.get(representative, preprocess_text(representative))
    representative_norm = alias2dedup_norm.get(representative, normalize_alias_for_dedup(representative))
    grouped_aliases = _group_aliases_by_kind(
        deduplicated_aliases,
        representative_norm=representative_norm,
        alias2dedup_norm=alias2dedup_norm,
    )
    ordered_aliases, selected_groups = _select_candidate_aliases(
        deduplicated_aliases,
        representative=representative,
        representative_norm=representative_norm,
        grouped_aliases=grouped_aliases,
        alias2language=alias2language,
        alias2dedup_norm=alias2dedup_norm,
        max_aliases=max_aliases,
        group_limits=group_limits,
        preferred_languages=preferred_languages,
    )
    candidate_text = _build_candidate_text(
        normalized_representative=normalized_representative,
        selected_groups=selected_groups,
        alias2normalized=alias2normalized,
    )
    language_counter = Counter(_normalize_language(value) for value in group_df.get("lang", pd.Series(dtype=str)).dropna().tolist())

    return {
        "CUI": str(cui),
        "semantic_type": str(semantic_type),
        "candidate_text": candidate_text,
        "representative_name": representative,
        "representative_name_normalized": normalized_representative,
        "selected_aliases": ordered_aliases,
        "selected_aliases_normalized": [alias2normalized.get(alias, preprocess_text(alias)) for alias in ordered_aliases],
        "selected_alias_count": int(len(ordered_aliases)),
        "num_aliases_raw": int(len(raw_aliases)),
        "num_aliases_kept": int(len(deduplicated_aliases)),
        "num_duplicates_removed": int(removed_duplicates),
        "num_long_removed": int(removed_long),
        "num_noisy_removed": int(removed_noisy),
        "languages": sorted(language_counter.keys()),
        "alias_length_threshold": int(alias_length_threshold),
    }


def _build_candidate_context_rows_for_subset(
    subset_df: pd.DataFrame,
    alias_length_threshold: int,
    max_aliases: int,
    group_limits: Dict[str, int],
    preferred_languages: list[str],
    allowed_languages: list[str],
) -> list[Dict]:
    """Build candidate-context cache rows for one vocabulary subset."""
    rows = []
    grouped = subset_df.groupby(["CUI", "semantic_type"], sort=False, dropna=False)
    for (cui, semantic_type), group_df in grouped:
        row = _build_candidate_context_row(
            cui,
            semantic_type,
            group_df,
            alias_length_threshold=alias_length_threshold,
            max_aliases=max_aliases,
            group_limits=group_limits,
            preferred_languages=preferred_languages,
            allowed_languages=allowed_languages,
        )
        if row is not None:
            rows.append(row)
    return rows


def build_candidate_context_cache(
    vocab_df: pd.DataFrame,
    *,
    alias_length_threshold: int | None = None,
    max_aliases: int = 8,
    group_limits: Dict[str, int] | None = None,
    preferred_languages: list[str] | None = None,
    allowed_languages: list[str] | None = None,
    num_workers: int = 1,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Build candidate-context rows from vocabulary aliases."""
    required_columns = {"CUI", "semantic_type", "concept_name"}
    missing_columns = required_columns.difference(vocab_df.columns)
    if missing_columns:
        raise ValueError(f"vocab_df is missing required columns: {sorted(missing_columns)}")

    length_stats = estimate_alias_length_threshold(vocab_df["concept_name"].tolist())
    if alias_length_threshold is None:
        alias_length_threshold = int(length_stats["threshold"])

    group_limits = {**DEFAULT_GROUP_LIMITS, **(group_limits or {})}
    preferred_languages = [_normalize_language(language) for language in (preferred_languages or [])]
    allowed_languages = [_normalize_language(language) for language in (allowed_languages or [])]
    rows = []

    logger.info(
        "Building candidate context cache: vocab_rows=%d, alias_length_threshold=%d, max_aliases=%d, num_workers=%d",
        len(vocab_df),
        alias_length_threshold,
        max_aliases,
        max(int(num_workers), 1),
    )
    num_workers = max(int(num_workers), 1)
    semantic_type_subsets = [
        subset_df.copy()
        for _, subset_df in vocab_df.groupby("semantic_type", sort=False, dropna=False)
    ]
    worker_kwargs = {
        "alias_length_threshold": int(alias_length_threshold),
        "max_aliases": int(max_aliases),
        "group_limits": group_limits,
        "preferred_languages": preferred_languages,
        "allowed_languages": allowed_languages,
    }
    if num_workers > 1 and len(semantic_type_subsets) > 1:
        with ProcessPoolExecutor(max_workers=min(num_workers, len(semantic_type_subsets))) as executor:
            row_batches = executor.map(
                _build_candidate_context_rows_for_subset,
                semantic_type_subsets,
                repeat(worker_kwargs["alias_length_threshold"]),
                repeat(worker_kwargs["max_aliases"]),
                repeat(worker_kwargs["group_limits"]),
                repeat(worker_kwargs["preferred_languages"]),
                repeat(worker_kwargs["allowed_languages"]),
            )
            for batch_rows in row_batches:
                rows.extend(batch_rows)
    else:
        for subset_df in semantic_type_subsets:
            rows.extend(
                _build_candidate_context_rows_for_subset(
                    subset_df,
                    **worker_kwargs,
                )
            )

    cache_df = pd.DataFrame(rows).sort_values(["semantic_type", "CUI"], kind="stable").reset_index(drop=True)
    metadata = {
        **length_stats,
        "threshold": int(alias_length_threshold),
        "num_candidate_rows": int(len(cache_df)),
        "max_aliases": int(max_aliases),
        "preferred_languages": preferred_languages,
        "allowed_languages": allowed_languages,
    }
    logger.info(
        "Built candidate context cache: num_rows=%d, threshold=%d",
        len(cache_df),
        alias_length_threshold,
    )
    return cache_df, metadata
