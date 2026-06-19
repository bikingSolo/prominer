"""Mention context construction for nested entity reranking."""

import logging
import re
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


BASE_ENTITY_COLUMNS = [
    "document_id",
    "text",
    "entity_type",
    "spans",
    "UMLS_CUI",
]

CONTEXTUALIZED_MENTION_COLUMN = "contextualized_mention_text"

CONTEXTUALIZED_MENTION_COLUMNS = [
    "mention_text",
    "left_context_text",
    "right_context_text",
    "context_text",
    "context_mode",
    CONTEXTUALIZED_MENTION_COLUMN,
    "context_source",
]


def validate_entity_columns(entities_df: pd.DataFrame) -> None:
    """Validate required entity dataframe columns."""
    missing_columns = sorted(set(BASE_ENTITY_COLUMNS).difference(entities_df.columns))
    if missing_columns:
        raise ValueError(f"entities_df is missing required columns: {missing_columns}")


def format_contextualized_mention(
    mention_text: str,
    *,
    context_format: str,
    sep_token: str = "[SEP]",
    context_text: str = "",
    left_context_text: str = "",
    right_context_text: str = "",
) -> str:
    """Format a mention with surrounding nested-entity context."""
    mention_text = str(mention_text or "").strip()
    left_context_text = str(left_context_text or "").strip()
    right_context_text = str(right_context_text or "").strip()
    context_text = str(context_text or "").strip()

    if not any((left_context_text, right_context_text, context_text)):
        return mention_text

    has_window_context = bool(left_context_text or right_context_text)

    if context_format == "sep_token":
        if has_window_context:
            parts = [part for part in (left_context_text, mention_text, right_context_text) if part]
            return f" {sep_token} ".join(parts).strip()
        return f"{mention_text} {sep_token} {context_text}".strip()
    if context_format == "explicit_markers":
        if has_window_context:
            parts = [part for part in (left_context_text, mention_text, right_context_text) if part]
            return " | ".join(parts).strip()
        return " | ".join(part for part in (mention_text, context_text) if part).strip()
    raise ValueError(f"Unsupported context_format={context_format!r}")


def _build_nested_context_details_for_row(doc_rows: pd.DataFrame, row) -> dict[str, str]:
    row_start = int(row["_span_start"])
    row_end = int(row["_span_end"])
    enclosing_rows = doc_rows[
        (doc_rows["_spans_str"] != row["_spans_str"])
        & (doc_rows["_span_start"] <= row_start)
        & (doc_rows["_span_end"] >= row_end)
    ].copy()

    if enclosing_rows.empty:
        return {
            "left_context_text": "",
            "right_context_text": "",
            "context_text": "",
        }

    enclosing_rows["_span_width"] = enclosing_rows["_span_end"] - enclosing_rows["_span_start"]
    left_candidates = enclosing_rows[enclosing_rows["_span_start"] < row_start]
    right_candidates = enclosing_rows[enclosing_rows["_span_end"] > row_end]

    def _select_longest_candidate(candidates_df: pd.DataFrame):
        if candidates_df.empty:
            return None
        return candidates_df.sort_values(
            by=["_span_width", "_span_start", "_span_end", "_mention_text_str"],
            ascending=[False, True, False, True],
            kind="stable",
        ).iloc[0]

    left_row = _select_longest_candidate(left_candidates)
    right_row = _select_longest_candidate(right_candidates)

    left_context_text = ""
    right_context_text = ""
    if left_row is not None:
        left_context_text = str(left_row["_mention_text_str"]).strip()
    if right_row is not None:
        right_context_text = str(right_row["_mention_text_str"]).strip()

    if left_row is not None and right_row is not None:
        if (
            str(left_row["_spans_str"]) == str(right_row["_spans_str"])
            or left_context_text == right_context_text
        ):
            left_context_text = ""
            right_context_text = right_context_text or left_context_text

    context_text = " ".join(part for part in (left_context_text, right_context_text) if part).strip()
    return {
        "left_context_text": left_context_text,
        "right_context_text": right_context_text,
        "context_text": context_text,
    }


def build_nested_context_details_map(
    entities_df: pd.DataFrame,
    *,
    drop_non_nested: bool = False,
) -> dict[tuple[str, str], dict[str, str]]:
    """Build nested-entity context details for each mention."""
    validate_entity_columns(entities_df)

    result_df = entities_df.copy()
    result_df["_document_id_str"] = result_df["document_id"].astype(str)
    result_df["_spans_str"] = result_df["spans"].astype(str)
    result_df["_mention_text_str"] = result_df["text"].fillna("").astype(str).str.strip()
    result_df["_span_bounds"] = result_df["_spans_str"].map(parse_span_bounds)
    result_df["_span_start"] = result_df["_span_bounds"].map(lambda bounds: bounds[0])
    result_df["_span_end"] = result_df["_span_bounds"].map(lambda bounds: bounds[1])

    context_map: dict[tuple[str, str], dict[str, str]] = {}
    for document_id, doc_df in result_df.groupby("_document_id_str", sort=False):
        doc_rows = doc_df.sort_values(
            by=["_span_start", "_span_end", "_mention_text_str"],
            ascending=[True, False, True],
            kind="stable",
        )

        for _, row in doc_rows.iterrows():
            details = _build_nested_context_details_for_row(doc_rows, row)
            if drop_non_nested and not details["context_text"]:
                context_map[(document_id, row["_spans_str"])] = {
                    "left_context_text": "",
                    "right_context_text": "",
                    "context_text": "",
                }
                continue
            context_map[(document_id, row["_spans_str"])] = details

    return context_map


def parse_span_bounds(spans_value: str) -> tuple[int, int]:
    """Parse a serialized span into integer bounds."""
    parts = str(spans_value).split(",")
    span_pairs = []
    for part in parts:
        start_str, end_str = part.split("-")
        span_pairs.append((int(start_str), int(end_str)))
    min_start = min(start for start, _ in span_pairs)
    max_end = max(end for _, end in span_pairs)
    return min_start, max_end


def infer_language_from_document_id(document_id: str) -> str:
    """Infer document language from its identifier."""
    doc_id = str(document_id)
    if doc_id.endswith("_ru"):
        return "ru"
    if doc_id.endswith("_en"):
        return "en"
    raise ValueError(f"Cannot infer language from document_id={document_id!r}")


def build_default_text_path(
    document_id: str,
    *,
    split_name: str,
    texts_root: str | Path = "data/texts",
) -> Path:
    """Build the expected raw-text path for a document."""
    language = infer_language_from_document_id(document_id)
    return Path(texts_root) / language / str(split_name) / f"{document_id}.txt"


def load_document_text(
    document_id: str,
    *,
    split_name: str,
    texts_root: str | Path = "data/texts",
    text_dir_template: str | None = None,
    text_cache: dict[str, str] | None = None,
) -> str:
    """Load raw document text from disk."""
    if text_cache is not None and document_id in text_cache:
        return text_cache[document_id]

    if text_dir_template:
        language = infer_language_from_document_id(document_id)
        text_path = Path(
            str(text_dir_template).format(
                texts_root=str(texts_root),
                lang=language,
                split=str(split_name),
                document_id=str(document_id),
            )
        )
    else:
        text_path = build_default_text_path(
            document_id=document_id,
            split_name=split_name,
            texts_root=texts_root,
        )

    if not text_path.exists():
        raise FileNotFoundError(
            f"Document text file not found for document_id={document_id!r}: {text_path}"
        )

    document_text = text_path.read_text(encoding="utf-8")
    if text_cache is not None:
        text_cache[document_id] = document_text
    return document_text


def build_text_window_context(
    document_text: str,
    *,
    spans_value: str,
    left_window_words: int,
    right_window_words: int,
) -> tuple[str, str]:
    """Build a text-window context string around a mention."""
    span_start, span_end = parse_span_bounds(spans_value)
    left_window_words = max(0, int(left_window_words))
    right_window_words = max(0, int(right_window_words))

    left_context_text = document_text[:span_start]
    right_context_text = document_text[span_end:]

    token_pattern = re.compile(r"\S+")
    left_tokens = token_pattern.findall(left_context_text)
    right_tokens = token_pattern.findall(right_context_text)

    left_context = " ".join(left_tokens[-left_window_words:]).strip() if left_window_words else ""
    right_context = " ".join(right_tokens[:right_window_words]).strip() if right_window_words else ""
    left_context = re.sub(r"[\(\[\{«\"']+\s*$", "", left_context).strip()
    right_context = re.sub(r"^\s*[\)\]\}»\"']+", "", right_context).strip()
    return left_context, right_context


def _contains_greek_letters(text: str) -> bool:
    return bool(re.search(r"[α-ωΑ-Ω]", str(text or "")))


def should_add_hybrid_window_context(raw_mention_text: str, *, entity_type: str | None = None) -> bool:
    """Decide whether a mention should receive window context."""
    mention_text = str(raw_mention_text or "").strip()
    if not mention_text:
        return False

    token_count = len(re.findall(r"\S+", mention_text))
    if token_count != 1:
        return False

    char_len = len(mention_text)
    contains_latin = bool(re.search(r"[A-Za-z]", mention_text))
    contains_digit = bool(re.search(r"\d", mention_text))
    contains_upper = bool(re.search(r"[A-ZА-ЯЁ]", mention_text))
    contains_symbol = bool(re.search(r"[/\-+]", mention_text))
    contains_greek = _contains_greek_letters(mention_text)
    alpha_chars = re.findall(r"[A-Za-zА-Яа-яЁё]", mention_text)
    upper_chars = re.findall(r"[A-ZА-ЯЁ]", mention_text)
    allowed_form = bool(re.fullmatch(r"[A-Za-zА-Яа-яЁё0-9/\-+α-ωΑ-Ω]+", mention_text))
    entity_type = str(entity_type or "").strip().upper()

    abbreviation_like = (
        2 <= char_len <= 12
        and allowed_form
        and (
            (contains_upper and len(upper_chars) >= max(1, len(alpha_chars) - 1))
            or contains_latin
            or contains_digit
            or contains_symbol
            or contains_greek
        )
    )
    short_symbolic = 2 <= char_len <= 8 and (contains_latin or contains_digit or contains_symbol or contains_greek)
    ultra_short_upper = 2 <= char_len <= 4 and contains_upper
    chemistry_symbolic = entity_type == "CHEM" and 2 <= char_len <= 10 and (contains_greek or contains_digit or contains_symbol)
    return bool(abbreviation_like or short_symbolic or ultra_short_upper or chemistry_symbolic)


def _resolve_hybrid_window_sizes(
    *,
    hybrid_window_words: int | None = None,
    hybrid_left_window_words: int | None = None,
    hybrid_right_window_words: int | None = None,
) -> tuple[int, int]:
    if hybrid_window_words is not None:
        return max(0, int(hybrid_window_words)), max(0, int(hybrid_window_words))
    left_window = 4 if hybrid_left_window_words is None else max(0, int(hybrid_left_window_words))
    right_window = 4 if hybrid_right_window_words is None else max(0, int(hybrid_right_window_words))
    return left_window, right_window


def build_contextualized_mentions(
    entities_df: pd.DataFrame,
    *,
    context_mode: str,
    context_format: str,
    sep_token: str = "[SEP]",
    drop_non_nested: bool = False,
    split_name: str | None = None,
    texts_root: str | Path = "data/texts",
    text_dir_template: str | None = None,
    window_words: int | None = None,
    left_window_words: int | None = None,
    right_window_words: int | None = None,
    hybrid_window_words: int | None = None,
    hybrid_left_window_words: int | None = None,
    hybrid_right_window_words: int | None = None,
    window_chars: int | None = None,
    left_window_chars: int | None = None,
    right_window_chars: int | None = None,
    text_preprocessor=None,
) -> pd.DataFrame:
    """Add contextualized mention strings to an entity dataframe."""
    validate_entity_columns(entities_df)

    result_df = entities_df.copy()
    result_df["mention_text"] = result_df["text"].astype(str)
    result_df["left_context_text"] = ""
    result_df["right_context_text"] = ""

    if context_mode == "nested_entities":
        context_map = build_nested_context_details_map(
            result_df,
            drop_non_nested=drop_non_nested,
        )
        result_df["left_context_text"] = [
            context_map.get((str(document_id), str(spans)), {}).get("left_context_text", "")
            for document_id, spans in zip(result_df["document_id"], result_df["spans"])
        ]
        result_df["right_context_text"] = [
            context_map.get((str(document_id), str(spans)), {}).get("right_context_text", "")
            for document_id, spans in zip(result_df["document_id"], result_df["spans"])
        ]
        result_df["context_text"] = [
            context_map.get((str(document_id), str(spans)), {}).get("context_text", "")
            for document_id, spans in zip(result_df["document_id"], result_df["spans"])
        ]
        result_df["context_source"] = "nested_entities"

    elif context_mode == "hybrid":
        if split_name is None:
            raise ValueError("split_name must be provided for context_mode='hybrid'")
        context_map = build_nested_context_details_map(
            result_df,
            drop_non_nested=drop_non_nested,
        )
        hybrid_left_window_words, hybrid_right_window_words = _resolve_hybrid_window_sizes(
            hybrid_window_words=hybrid_window_words,
            hybrid_left_window_words=hybrid_left_window_words,
            hybrid_right_window_words=hybrid_right_window_words,
        )
        text_cache: dict[str, str] = {}
        left_context_texts = []
        right_context_texts = []
        context_texts = []
        context_sources = []
        for document_id, spans, raw_mention_text, entity_type in zip(
            result_df["document_id"],
            result_df["spans"],
            result_df["text"],
            result_df["entity_type"],
        ):
            details = context_map.get(
                (str(document_id), str(spans)),
                {"left_context_text": "", "right_context_text": "", "context_text": ""},
            )
            left_context_text = str(details.get("left_context_text", "")).strip()
            right_context_text = str(details.get("right_context_text", "")).strip()
            context_text = str(details.get("context_text", "")).strip()
            context_source = "hybrid_nested_entities"

            if not context_text:
                if should_add_hybrid_window_context(str(raw_mention_text), entity_type=str(entity_type)):
                    document_text = load_document_text(
                        str(document_id),
                        split_name=split_name,
                        texts_root=texts_root,
                        text_dir_template=text_dir_template,
                        text_cache=text_cache,
                    )
                    left_context_text, right_context_text = build_text_window_context(
                        document_text=document_text,
                        spans_value=str(spans),
                        left_window_words=hybrid_left_window_words,
                        right_window_words=hybrid_right_window_words,
                    )
                    context_text = " ".join(
                        part for part in (left_context_text, right_context_text) if part
                    ).strip()
                    context_source = "hybrid_text_window" if context_text else "hybrid_none"
                else:
                    context_source = "hybrid_none"

            left_context_texts.append(left_context_text)
            right_context_texts.append(right_context_text)
            context_texts.append(context_text)
            context_sources.append(context_source)
        result_df["left_context_text"] = left_context_texts
        result_df["right_context_text"] = right_context_texts
        result_df["context_text"] = context_texts
        result_df["context_source"] = context_sources

    elif context_mode == "text_window":
        if split_name is None:
            raise ValueError("split_name must be provided for context_mode='text_window'")
        if window_words is not None:
            left_window_words = int(window_words)
            right_window_words = int(window_words)
        elif window_chars is not None:
            logger.warning(
                "window_chars is deprecated for text_window context and treated as a word count fallback"
            )
            left_window_words = int(window_chars)
            right_window_words = int(window_chars)

        if left_window_words is None and left_window_chars is not None:
            logger.warning(
                "left_window_chars is deprecated for text_window context and treated as a word count fallback"
            )
            left_window_words = int(left_window_chars)
        if right_window_words is None and right_window_chars is not None:
            logger.warning(
                "right_window_chars is deprecated for text_window context and treated as a word count fallback"
            )
            right_window_words = int(right_window_chars)

        left_window_words = 0 if left_window_words is None else int(left_window_words)
        right_window_words = 0 if right_window_words is None else int(right_window_words)

        text_cache: dict[str, str] = {}
        left_context_texts = []
        right_context_texts = []
        context_texts = []
        for document_id, spans in zip(result_df["document_id"], result_df["spans"]):
            document_text = load_document_text(
                str(document_id),
                split_name=split_name,
                texts_root=texts_root,
                text_dir_template=text_dir_template,
                text_cache=text_cache,
            )
            left_context_text, right_context_text = build_text_window_context(
                document_text=document_text,
                spans_value=str(spans),
                left_window_words=left_window_words,
                right_window_words=right_window_words,
            )
            left_context_texts.append(left_context_text)
            right_context_texts.append(right_context_text)
            context_texts.append(" ".join(part for part in (left_context_text, right_context_text) if part).strip())

        result_df["left_context_text"] = left_context_texts
        result_df["right_context_text"] = right_context_texts
        result_df["context_text"] = context_texts
        result_df["context_source"] = "text_window"

    else:
        raise ValueError(f"Unsupported context_mode={context_mode!r}")

    if text_preprocessor is not None:
        result_df["mention_text"] = result_df["mention_text"].map(text_preprocessor)
        result_df["left_context_text"] = result_df["left_context_text"].map(text_preprocessor)
        result_df["right_context_text"] = result_df["right_context_text"].map(text_preprocessor)
        result_df["context_text"] = result_df["context_text"].map(text_preprocessor)

    result_df["context_mode"] = str(context_mode)
    result_df[CONTEXTUALIZED_MENTION_COLUMN] = [
        format_contextualized_mention(
            mention_text=mention_text,
            context_format=context_format,
            sep_token=sep_token,
            context_text=context_text,
            left_context_text=left_context_text,
            right_context_text=right_context_text,
        )
        for mention_text, left_context_text, right_context_text, context_text in zip(
            result_df["mention_text"],
            result_df["left_context_text"],
            result_df["right_context_text"],
            result_df["context_text"],
        )
    ]

    logger.info(
        "Prepared contextualized mentions: context_mode=%s, context_format=%s, num_rows=%d",
        context_mode,
        context_format,
        len(result_df),
    )
    return result_df


def summarize_context_coverage(
    contextualized_df: pd.DataFrame,
    *,
    contextualized_mention_column: str = CONTEXTUALIZED_MENTION_COLUMN,
) -> dict[str, int]:
    """Summarize context availability for contextualized mentions."""
    context_non_empty = contextualized_df["context_text"].fillna("").astype(str).str.len() > 0
    mention_changed = (
        contextualized_df[contextualized_mention_column].fillna("").astype(str)
        != contextualized_df["mention_text"].fillna("").astype(str)
    )
    return {
        "num_rows": int(len(contextualized_df)),
        "num_non_empty_context": int(context_non_empty.sum()),
        "num_contextualized_mentions": int(mention_changed.sum()),
    }
