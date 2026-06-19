"""Training pair builders for dense retriever fine-tuning."""

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from ..core.dense import get_dense_topk_batched


logger = logging.getLogger(__name__)


TRAINING_PAIR_COLUMNS = [
    "mention_text",
    "concept_name",
    "CUI",
    "entity_type",
]
RAW_MENTION_COLUMN = "raw_mention_text"


def _fingerprint_dataframe(df: pd.DataFrame, *, columns: list[str] | None = None) -> str:
    """Hash dataframe content for cache invalidation."""
    fingerprint_df = df if columns is None else df[[column for column in columns if column in df.columns]].copy()
    hashed = pd.util.hash_pandas_object(fingerprint_df, index=True).values.tobytes()
    digest = hashlib.sha256()
    digest.update("|".join(fingerprint_df.columns.tolist()).encode("utf-8"))
    digest.update(str(fingerprint_df.shape).encode("utf-8"))
    digest.update(hashed)
    return digest.hexdigest()


def _build_hard_negative_cache_key(
    train_pairs_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    *,
    model_id: str,
    mention_column: str,
    raw_mention_column: str | None,
    query_batch_size: int,
    dense_vocab_batch_size: int,
    st_encode_batch_size: int,
    num_hard_negatives: int,
    hard_negative_deduplicate_by_cui: bool,
    hard_negative_skip_topk: int,
    cui_overfetch_factor: int,
) -> str:
    """Build a stable cache key for mined hard negatives."""
    cache_payload = {
        "model_id": str(model_id),
        "mention_column": str(mention_column),
        "raw_mention_column": None if raw_mention_column is None else str(raw_mention_column),
        "query_batch_size": int(query_batch_size),
        "dense_vocab_batch_size": int(dense_vocab_batch_size),
        "st_encode_batch_size": int(st_encode_batch_size),
        "num_hard_negatives": int(num_hard_negatives),
        "hard_negative_deduplicate_by_cui": bool(hard_negative_deduplicate_by_cui),
        "hard_negative_skip_topk": int(hard_negative_skip_topk),
        "cui_overfetch_factor": int(cui_overfetch_factor),
        "train_pairs_fingerprint": _fingerprint_dataframe(train_pairs_df),
        "vocab_fingerprint": _fingerprint_dataframe(
            vocab_df,
            columns=["concept_name", "CUI", "semantic_type", "lang"],
        ),
    }
    return hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()


def _build_positive_vocab(vocab_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare unique positive concept names from the vocabulary."""
    positive_vocab = (
        vocab_df[["concept_name", "CUI", "semantic_type"]]
        .dropna()
        .copy()
    )
    positive_vocab["concept_name"] = positive_vocab["concept_name"].astype(str)
    positive_vocab["CUI"] = positive_vocab["CUI"].astype(str)
    positive_vocab["semantic_type"] = positive_vocab["semantic_type"].astype(str)
    positive_vocab = positive_vocab[positive_vocab["concept_name"].str.len() > 0].copy()
    positive_vocab = positive_vocab.rename(columns={"semantic_type": "entity_type"})
    positive_vocab = positive_vocab.sort_values(
        ["CUI", "entity_type", "concept_name"],
        kind="mergesort",
    ).drop_duplicates(
        subset=["concept_name", "CUI", "entity_type"],
        keep="first",
    )
    return positive_vocab


def build_dense_training_pairs(
    entities_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    *,
    mention_column: str = "text",
    raw_mention_column: str | None = None,
) -> pd.DataFrame:
    """Build positive dense retriever training pairs."""
    required_entity_columns = {mention_column, "UMLS_CUI", "entity_type"}
    if raw_mention_column is not None:
        required_entity_columns.add(raw_mention_column)
    missing_entity_columns = required_entity_columns.difference(entities_df.columns)
    if missing_entity_columns:
        raise ValueError(f"entities_df is missing required columns: {sorted(missing_entity_columns)}")

    positive_vocab = _build_positive_vocab(vocab_df=vocab_df)

    source_columns = [mention_column, "UMLS_CUI", "entity_type"]
    if raw_mention_column is not None:
        source_columns.append(raw_mention_column)
    train_pairs_df = entities_df[source_columns].dropna(subset=[mention_column, "UMLS_CUI", "entity_type"]).copy()
    train_pairs_df["mention_text"] = train_pairs_df[mention_column].astype(str)
    train_pairs_df["CUI"] = train_pairs_df["UMLS_CUI"].astype(str)
    train_pairs_df["entity_type"] = train_pairs_df["entity_type"].astype(str)
    if raw_mention_column is not None:
        train_pairs_df[RAW_MENTION_COLUMN] = train_pairs_df[raw_mention_column].astype(str)
    train_pairs_df = train_pairs_df[
        (train_pairs_df["CUI"] != "CUILESS")
        & (train_pairs_df["mention_text"].str.len() > 0)
    ].copy()

    pair_columns = ["mention_text", "CUI", "entity_type"]
    if raw_mention_column is not None:
        pair_columns.append(RAW_MENTION_COLUMN)

    train_pairs_df = train_pairs_df[pair_columns].drop_duplicates(
        subset=["mention_text", "CUI", "entity_type"],
        keep="first",
    )

    train_pairs_df = train_pairs_df.merge(
        positive_vocab,
        on=["CUI", "entity_type"],
        how="inner",
        validate="many_to_many",
    )
    output_columns = TRAINING_PAIR_COLUMNS[:]
    if raw_mention_column is not None:
        output_columns.append(RAW_MENTION_COLUMN)
    train_pairs_df = train_pairs_df[output_columns].drop_duplicates(
        subset=output_columns,
        keep="first",
    ).reset_index(drop=True)

    logger.info(
        "Prepared dense training pairs from column=%s: num_pairs=%d, num_unique_mentions=%d, num_unique_cuis=%d, num_entity_types=%d",
        mention_column,
        len(train_pairs_df),
        train_pairs_df["mention_text"].nunique(),
        train_pairs_df["CUI"].nunique(),
        train_pairs_df["entity_type"].nunique(),
    )
    return train_pairs_df


def _build_dense_training_examples_from_pairs(
    train_pairs_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    *,
    st_model,
    query_batch_size: int,
    dense_vocab_batch_size: int,
    st_encode_batch_size: int,
    num_hard_negatives: int = 0,
    hard_negative_deduplicate_by_cui: bool = True,
    hard_negative_skip_topk: int = 0,
    cui_overfetch_factor: int = 8,
) -> pd.DataFrame:
    """Add optional offline hard negatives to dense training pairs."""
    positive_vocab = _build_positive_vocab(vocab_df=vocab_df)
    num_hard_negatives = int(num_hard_negatives)
    hard_negative_skip_topk = max(int(hard_negative_skip_topk), 0)
    if num_hard_negatives <= 0:
        logger.info("NUM_HARD_NEGATIVES=%d, using pair-only training data", num_hard_negatives)
        return train_pairs_df

    negative_columns = [f"hard_negative_{idx}" for idx in range(1, num_hard_negatives + 1)]
    train_examples_df = train_pairs_df.copy()
    for column in negative_columns:
        train_examples_df[column] = None

    entity_types = sorted(train_examples_df["entity_type"].dropna().unique().tolist())
    retrieval_base_k = max(
        (num_hard_negatives + hard_negative_skip_topk) * max(int(cui_overfetch_factor), 1),
        num_hard_negatives + hard_negative_skip_topk + 8,
    )
    missing_negative_count = 0

    logger.info(
        "Mining offline hard negatives: num_examples=%d, num_entity_types=%d, num_hard_negatives=%d, skip_topk=%d, deduplicate_by_cui=%s",
        len(train_examples_df),
        len(entity_types),
        num_hard_negatives,
        hard_negative_skip_topk,
        hard_negative_deduplicate_by_cui,
    )

    for entity_type in entity_types:
        subset_idx = train_examples_df.index[train_examples_df["entity_type"] == entity_type].tolist()
        subset_df = train_examples_df.loc[subset_idx].reset_index(drop=True)
        subset_vocab = positive_vocab[positive_vocab["entity_type"] == entity_type].reset_index(drop=True)
        vocab_names = subset_vocab["concept_name"].astype(str).values
        vocab_cuis = subset_vocab["CUI"].astype(str).values

        if len(subset_vocab) == 0:
            logger.warning("Skipping hard-negative mining for entity_type=%s because vocab subset is empty", entity_type)
            continue

        _, dense_indices = get_dense_topk_batched(
            query_names=subset_df["mention_text"].astype(str).values,
            vocab_names=vocab_names,
            vocab_cuis=vocab_cuis,
            st_model=st_model,
            base_k=min(len(vocab_names), retrieval_base_k),
            query_batch_size=query_batch_size,
            vocab_batch_size=dense_vocab_batch_size,
            st_encode_batch_size=st_encode_batch_size,
            deduplicate_by_cui=hard_negative_deduplicate_by_cui,
            cui_overfetch_factor=cui_overfetch_factor,
            show_progress=True,
        )

        for local_row_idx, global_row_idx in enumerate(subset_idx):
            gold_cui = str(train_examples_df.at[global_row_idx, "CUI"])
            mention_text = str(train_examples_df.at[global_row_idx, "mention_text"])
            positive_name = str(train_examples_df.at[global_row_idx, "concept_name"])
            forbidden_normalized_texts = {positive_name}
            if RAW_MENTION_COLUMN in train_examples_df.columns:
                raw_mention_text = train_examples_df.at[global_row_idx, RAW_MENTION_COLUMN]
                if pd.notna(raw_mention_text):
                    forbidden_normalized_texts.add(str(raw_mention_text))
            else:
                forbidden_normalized_texts.add(mention_text)
            hard_negative_names = []
            seen_normalized_names = set(forbidden_normalized_texts)
            num_skipped_valid_negatives = 0

            for candidate_idx in dense_indices[local_row_idx]:
                candidate_idx = int(candidate_idx)
                if candidate_idx < 0:
                    continue

                candidate_cui = str(vocab_cuis[candidate_idx])
                candidate_name = str(vocab_names[candidate_idx])
                if candidate_cui == gold_cui:
                    continue
                if not candidate_name:
                    continue
                if candidate_name in seen_normalized_names:
                    continue
                if num_skipped_valid_negatives < hard_negative_skip_topk:
                    num_skipped_valid_negatives += 1
                    seen_normalized_names.add(candidate_name)
                    continue

                seen_normalized_names.add(candidate_name)
                hard_negative_names.append(candidate_name)
                if len(hard_negative_names) == num_hard_negatives:
                    break

            if not hard_negative_names:
                missing_negative_count += num_hard_negatives
                continue

            if len(hard_negative_names) < num_hard_negatives:
                missing_negative_count += num_hard_negatives - len(hard_negative_names)
                hard_negative_names.extend([hard_negative_names[-1]] * (num_hard_negatives - len(hard_negative_names)))

            for column, negative_name in zip(negative_columns, hard_negative_names):
                train_examples_df.at[global_row_idx, column] = negative_name

    logger.info(
        "Finished offline hard-negative mining: total_missing_negative_slots=%d",
        missing_negative_count,
    )
    return train_examples_df


def build_dense_training_examples(
    entities_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    *,
    mention_column: str = "text",
    raw_mention_column: str | None = None,
    st_model,
    model_id: str | None = None,
    query_batch_size: int,
    dense_vocab_batch_size: int,
    st_encode_batch_size: int,
    num_hard_negatives: int = 0,
    hard_negative_deduplicate_by_cui: bool = True,
    hard_negative_skip_topk: int = 0,
    cui_overfetch_factor: int = 8,
    hard_negative_cache_dir: str | None = None,
) -> pd.DataFrame:
    """Build dense retriever training examples with metadata."""
    train_pairs_df = build_dense_training_pairs(
        entities_df=entities_df,
        vocab_df=vocab_df,
        mention_column=mention_column,
        raw_mention_column=raw_mention_column,
    )
    num_hard_negatives = int(num_hard_negatives)
    hard_negative_cache_path = None
    if num_hard_negatives > 0 and hard_negative_cache_dir:
        cache_dir = Path(hard_negative_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = _build_hard_negative_cache_key(
            train_pairs_df=train_pairs_df,
            vocab_df=vocab_df,
            model_id=str(model_id or ""),
            mention_column=mention_column,
            raw_mention_column=raw_mention_column,
            query_batch_size=query_batch_size,
            dense_vocab_batch_size=dense_vocab_batch_size,
            st_encode_batch_size=st_encode_batch_size,
            num_hard_negatives=num_hard_negatives,
            hard_negative_deduplicate_by_cui=hard_negative_deduplicate_by_cui,
            hard_negative_skip_topk=hard_negative_skip_topk,
            cui_overfetch_factor=cui_overfetch_factor,
        )
        hard_negative_cache_path = cache_dir / f"{cache_key}.parquet"
        if hard_negative_cache_path.exists():
            logger.info("Loading cached hard-negative training examples from %s", hard_negative_cache_path)
            return pd.read_parquet(hard_negative_cache_path)
    train_examples_df = _build_dense_training_examples_from_pairs(
        train_pairs_df=train_pairs_df,
        vocab_df=vocab_df,
        st_model=st_model,
        query_batch_size=query_batch_size,
        dense_vocab_batch_size=dense_vocab_batch_size,
        st_encode_batch_size=st_encode_batch_size,
        num_hard_negatives=num_hard_negatives,
        hard_negative_deduplicate_by_cui=hard_negative_deduplicate_by_cui,
        hard_negative_skip_topk=hard_negative_skip_topk,
        cui_overfetch_factor=cui_overfetch_factor,
    )
    if hard_negative_cache_path is not None:
        train_examples_df.to_parquet(hard_negative_cache_path, index=False)
        logger.info("Saved cached hard-negative training examples to %s", hard_negative_cache_path)
    return train_examples_df
