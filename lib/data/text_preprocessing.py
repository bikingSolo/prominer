"""Text normalization helpers shared by retrieval components."""

import re
import unicodedata
from typing import Iterable, List


QUOTE_TRANSLATION_TABLE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201A": "'",
        "\u201B": "'",
        "\u2032": "'",
        "\u00B4": "'",
        "\u0060": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u201E": '"',
        "\u201F": '"',
        "\u2033": '"',
        "\u00AB": '"',
        "\u00BB": '"',
    }
)


def preprocess_text(text: str) -> str:
    """Normalize a single text string for retrieval."""
    normalized_text = unicodedata.normalize("NFKC", str(text))
    normalized_text = normalized_text.translate(QUOTE_TRANSLATION_TABLE)
    normalized_text = normalized_text.lower()
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    # Remove repeated commas and trim terminal dots/commas while preserving
    # core punctuation such as hyphens, slashes, and numeric patterns.
    normalized_text = re.sub(r"\s*,\s*", ", ", normalized_text)
    normalized_text = re.sub(r"(,\s*){2,}", ", ", normalized_text)
    normalized_text = re.sub(r"[.,\s]+$", "", normalized_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
    return normalized_text


def preprocess_texts(texts: Iterable[str]) -> List[str]:
    """Normalize a list of text strings for retrieval."""
    return [preprocess_text(text) for text in texts]

