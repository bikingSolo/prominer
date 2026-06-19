"""Alias normalization and selection heuristics for candidate context."""

import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, Iterable

import pandas as pd


DEFAULT_GROUP_LIMITS = {
    "abbreviations": 2,
    "short_names": 2,
    "multi_word": 3,
    "long_variants": 1,
}
NOISY_CONTEXT_PATTERNS = (
    "&#x7c;",
    "@@",
)
NOISY_PIPE_TOKENS = (
    "radiology",
    "microbiology",
    "allergy",
    "serum",
    "urine",
    "plasma",
    "blood",
)
ONTOLOGY_SUFFIX_PATTERNS = (
    r"\s*\((?:body structure|finding|physical finding|substance|medication|disorder)\)\s*$",
    r"\s*-\s*chemical\s*\(substance\)\s*$",
)
SOFT_NOISY_SUFFIX_PATTERNS = (
    r"\s*\((?:product|qualifier value)\)\s*$",
)
HARD_NOISY_ALIAS_PATTERNS = (
    r"\bnos\b",
    r"\bnec\b",
    r"\bunspecified\b",
    r"\bnot elsewhere classified\b",
    r"\bto be specified in another part of the message\b",
    r"\bxxx\b",
)
LEADING_TRAILING_WRAPPER_PATTERNS = (
    r"^\s*structure of\s+",
    r"^\s*entire\s+",
    r"^\s*set of\s+",
    r"^\s*part of\s+",
)
PARENTHETICAL_DROP_PATTERNS = (
    r"\s*\((?:specimen|product|medicinal product|cell structure|body structure)\)\s*$",
)
TRAILING_ONTOLOGY_TERM_PATTERNS = (
    r"\s+structure\s*$",
    r"\s+structures\s*$",
)


@lru_cache(maxsize=200000)
def normalize_alias_for_dedup(text: str) -> str:
    """Normalize alias text for deterministic deduplication."""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.casefold()
    normalized = re.sub(r"[\u2018\u2019\u201A\u201B\u2032\u00B4\u0060]", "'", normalized)
    normalized = re.sub(r"[\u201C\u201D\u201E\u201F\u2033\u00AB\u00BB]", '"', normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s*([,;:/()\\[\\]{}+-])\s*", r"\1", normalized)
    normalized = re.sub(r"[.,;:\s]+$", "", normalized)
    return normalized


def estimate_alias_length_threshold(
    alias_texts: Iterable[str],
    *,
    quantile: float = 0.995,
    iqr_scale: float = 3.0,
    min_threshold: int = 96,
    max_threshold: int = 256,
) -> Dict[str, float]:
    """Estimate a robust maximum alias length threshold."""
    lengths = pd.Series([len(str(text or "").strip()) for text in alias_texts], dtype="float64")
    lengths = lengths[lengths > 0]
    if lengths.empty:
        return {
            "num_aliases": 0,
            "threshold": float(min_threshold),
            "quantile_value": float(min_threshold),
            "iqr_value": float(min_threshold),
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    q1 = float(lengths.quantile(0.25))
    q3 = float(lengths.quantile(0.75))
    quantile_value = float(lengths.quantile(quantile))
    iqr_value = float(q3 + iqr_scale * max(q3 - q1, 1.0))
    threshold = max(min_threshold, min(max_threshold, int(round(min(quantile_value, iqr_value)))))

    return {
        "num_aliases": int(len(lengths)),
        "threshold": float(threshold),
        "quantile_value": quantile_value,
        "iqr_value": iqr_value,
        "median": float(lengths.quantile(0.5)),
        "p95": float(lengths.quantile(0.95)),
        "p99": float(lengths.quantile(0.99)),
        "max": float(lengths.max()),
    }


@lru_cache(maxsize=200000)
def is_likely_abbreviation(text: str) -> bool:
    """Detect short aliases that behave like abbreviations."""
    text = str(text or "").strip()
    if not text:
        return False

    compact = re.sub(r"[\s./_-]+", "", text)
    if not compact:
        return False

    tokens = [token for token in re.split(r"[\s/,_-]+", text) if token]
    has_digits = any(char.isdigit() for char in compact)
    alpha_chars = [char for char in compact if char.isalpha()]

    if len(tokens) == 1 and len(compact) <= 6:
        return True
    if len(tokens) <= 2 and len(compact) <= 8 and has_digits:
        return True
    if len(tokens) <= 2 and len(compact) <= 10 and alpha_chars:
        max_token_len = max(len(token) for token in tokens)
        vowel_count = sum(char.lower() in "aeiouyаеёиоуыэюя" for char in alpha_chars)
        if max_token_len <= 5 or vowel_count <= max(1, len(alpha_chars) // 5):
            return True
    return False


@lru_cache(maxsize=200000)
def _classify_alias(text: str) -> str:
    stripped = str(text or "").strip()
    token_count = len([token for token in stripped.split() if token])
    if is_likely_abbreviation(stripped):
        return "abbreviations"
    if token_count <= 2 and len(stripped) <= 24:
        return "short_names"
    if token_count <= 5 and len(stripped) <= 80:
        return "multi_word"
    return "long_variants"


@lru_cache(maxsize=200000)
def _token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalize_alias_for_dedup(text)))


@lru_cache(maxsize=100000)
def _alias_similarity(left: str, right: str) -> float:
    left_norm = normalize_alias_for_dedup(left)
    right_norm = normalize_alias_for_dedup(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0

    left_tokens = _token_set(left_norm)
    right_tokens = _token_set(right_norm)
    token_jaccard = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    sequence_ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    return 0.5 * token_jaccard + 0.5 * sequence_ratio


@lru_cache(maxsize=200000)
def _has_cyrillic(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", str(text or "")))


@lru_cache(maxsize=200000)
def _has_mixed_scripts(text: str) -> bool:
    return _has_cyrillic(text) and bool(re.search(r"[A-Za-z]", str(text or "")))


@lru_cache(maxsize=200000)
def _is_all_caps_alias(text: str) -> bool:
    alpha_chars = [char for char in str(text or "") if char.isalpha()]
    return len(alpha_chars) >= 3 and all(char.isupper() for char in alpha_chars)


@lru_cache(maxsize=200000)
def _is_pipe_style_noise(text: str) -> bool:
    normalized_text = str(text or "").casefold()
    if "&#x7c;" in normalized_text:
        return True
    if "|" not in normalized_text:
        return False
    return any(token in normalized_text for token in NOISY_PIPE_TOKENS)


@lru_cache(maxsize=200000)
def _strip_ontology_suffix(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return cleaned
    previous = None
    while previous != cleaned:
        previous = cleaned
        for pattern in ONTOLOGY_SUFFIX_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


@lru_cache(maxsize=200000)
def _sanitize_alias_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    lowered = cleaned.casefold()
    if any(pattern in lowered for pattern in NOISY_CONTEXT_PATTERNS):
        return ""
    if _is_pipe_style_noise(cleaned):
        return ""
    cleaned = _strip_ontology_suffix(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for pattern in PARENTHETICAL_DROP_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    for pattern in LEADING_TRAILING_WRAPPER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    for pattern in TRAILING_ONTOLOGY_TERM_PATTERNS:
        if len(cleaned.split()) >= 2:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s*-\s*", "-", cleaned)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    cleaned = cleaned.strip(" ,;:")
    lowered = cleaned.casefold()
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in HARD_NOISY_ALIAS_PATTERNS):
        return ""
    if re.search(r"\b(specimen from|sample from|swab from)\b", lowered):
        return ""
    if re.search(r"\b(product containing|product$|containing product|medicinal product)\b", lowered):
        return ""
    if re.search(r"\b(dose form|only product|product in .+ dose form)\b", lowered):
        return ""
    if re.search(r"\b[a-z0-9_]+\.[a-z0-9_.]+\b", cleaned) and " mg/" not in lowered and " ml/" not in lowered:
        return ""
    if len(cleaned) <= 3 and not _has_cyrillic(cleaned) and not re.search(r"[A-Za-z]{2,}", cleaned):
        return ""
    return cleaned.strip()


@lru_cache(maxsize=200000)
def _has_soft_noisy_suffix(text: str) -> bool:
    cleaned = str(text or "").strip()
    return any(re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in SOFT_NOISY_SUFFIX_PATTERNS)


@lru_cache(maxsize=200000)
def _is_too_short_abbreviation(text: str) -> bool:
    compact = re.sub(r"[\s./_-]+", "", str(text or ""))
    alpha_chars = [char for char in compact if char.isalpha()]
    return is_likely_abbreviation(text) and len(alpha_chars) <= 3 and len(compact) <= 5


@lru_cache(maxsize=200000)
def _is_fragment_like_alias(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    lowered = normalized.casefold()
    if ";" not in lowered:
        return False
    tokens = [token.strip() for token in re.split(r"[;,:/]+", lowered) if token.strip()]
    if len(tokens) < 2:
        return False
    short_token_count = sum(len(token.split()) <= 2 for token in tokens)
    return short_token_count >= 2


@lru_cache(maxsize=200000)
def _alias_quality_penalty(text: str) -> tuple[int, int, int]:
    return (
        1 if _has_soft_noisy_suffix(text) else 0,
        1 if _is_fragment_like_alias(text) else 0,
        1 if _is_too_short_abbreviation(text) else 0,
    )


def _pick_representative(aliases: list[str]) -> str:
    return min(
        aliases,
        key=lambda alias: (
            0 if _has_cyrillic(alias) else 1,
            0 if not _has_mixed_scripts(alias) else 1,
            *_alias_quality_penalty(alias),
            0 if not _is_all_caps_alias(alias) else 1,
            0 if not is_likely_abbreviation(alias) else 1,
            len(alias),
            alias.casefold(),
        ),
    )


def _normalize_language(value: str | None) -> str:
    language = str(value or "").strip().upper()
    if language == "EN":
        language = "ENG"
    elif language == "RU":
        language = "RUS"
    return language or "UNK"


def _language_priority(language: str, preferred_languages: list[str]) -> tuple[int, str]:
    normalized_language = _normalize_language(language)
    try:
        priority = preferred_languages.index(normalized_language)
    except ValueError:
        priority = len(preferred_languages)
    return priority, normalized_language


def _select_diverse_aliases(
    aliases: list[str],
    *,
    limit: int,
    representative: str,
    preferred_languages: list[str] | None = None,
    alias_languages: dict[str, str] | None = None,
) -> list[str]:
    if limit <= 0:
        return []
    preferred_languages = [_normalize_language(language) for language in (preferred_languages or [])]
    alias_languages = alias_languages or {}

    remaining = [alias for alias in aliases if normalize_alias_for_dedup(alias) != normalize_alias_for_dedup(representative)]
    if not remaining:
        return []

    selected = []
    language_priority_cache = {
        alias: _language_priority(alias_languages.get(alias), preferred_languages)[0]
        for alias in remaining
    }
    quality_penalty_cache = {alias: _alias_quality_penalty(alias) for alias in remaining}
    while remaining and len(selected) < limit:
        if not selected:
            best_alias = max(
                remaining,
                key=lambda alias: (
                    -language_priority_cache[alias],
                    -quality_penalty_cache[alias][0],
                    -quality_penalty_cache[alias][1],
                    -quality_penalty_cache[alias][2],
                    1.0 - _alias_similarity(alias, representative),
                    -len(alias),
                    alias.casefold(),
                ),
            )
        else:
            best_alias = max(
                remaining,
                key=lambda alias: (
                    -language_priority_cache[alias],
                    -quality_penalty_cache[alias][0],
                    -quality_penalty_cache[alias][1],
                    -quality_penalty_cache[alias][2],
                    min(1.0 - _alias_similarity(alias, current) for current in selected + [representative]),
                    -sum(_alias_similarity(alias, current) for current in selected),
                    -len(alias),
                    alias.casefold(),
                ),
            )
        selected.append(best_alias)
        remaining.remove(best_alias)
    return selected
