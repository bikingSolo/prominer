"""Batch samplers for dense retriever fine-tuning."""

import logging
import random
from collections import deque

import pandas as pd
from torch.utils.data import BatchSampler

from .data import RAW_MENTION_COLUMN


logger = logging.getLogger(__name__)


class NoDuplicateCuiBatchSampler(BatchSampler):
    """Sample batches without duplicate positive CUIs."""
    def __init__(
        self,
        cuis,
        text_keys_per_example,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 0,
        deduplicate_by_text: bool = False,
    ):
        self.cuis = [str(cui) for cui in cuis]
        self.text_keys_per_example = [set(text_keys) for text_keys in text_keys_per_example]
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.deduplicate_by_text = bool(deduplicate_by_text)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.unique_cuis = list(dict.fromkeys(self.cuis))
        self.cui2indices = {}
        for idx, cui in enumerate(self.cuis):
            self.cui2indices.setdefault(cui, []).append(idx)
        self._cached_num_batches = None

    def _iter_batches(self):
        rng = random.Random(self.seed)

        groups = {}
        for cui, indices in self.cui2indices.items():
            shuffled_indices = indices[:]
            rng.shuffle(shuffled_indices)
            groups[cui] = deque(shuffled_indices)

        active_cuis = list(groups.keys())
        rng.shuffle(active_cuis)

        if not self.deduplicate_by_text:
            batch = []
            while active_cuis:
                next_active_cuis = []
                for cui in active_cuis:
                    indices = groups[cui]
                    if not indices:
                        continue
                    batch.append(indices.pop())
                    if indices:
                        next_active_cuis.append(cui)
                    if len(batch) == self.batch_size:
                        yield list(batch)
                        batch = []

                active_cuis = next_active_cuis
                rng.shuffle(active_cuis)

            if batch and not self.drop_last:
                yield list(batch)
            return

        batch = []
        batch_text_keys = set()

        while active_cuis:
            next_active_cuis = []
            deferred_cuis = []
            made_progress = False
            for cui in active_cuis:
                indices = groups[cui]
                if not indices:
                    continue
                selected_idx = None

                # Try each remaining example for this CUI at most once in the current pass.
                num_candidates = len(indices)
                for _ in range(num_candidates):
                    candidate_idx = indices.popleft()
                    candidate_text_keys = self.text_keys_per_example[candidate_idx]
                    if (not self.deduplicate_by_text) or candidate_text_keys.isdisjoint(batch_text_keys):
                        selected_idx = candidate_idx
                        break
                    indices.append(candidate_idx)

                if selected_idx is None:
                    deferred_cuis.append(cui)
                    continue

                batch.append(selected_idx)
                batch_text_keys.update(self.text_keys_per_example[selected_idx])
                made_progress = True
                if indices:
                    next_active_cuis.append(cui)
                if len(batch) == self.batch_size:
                    yield list(batch)
                    batch = []
                    batch_text_keys = set()

            if batch and deferred_cuis and not made_progress:
                if not self.drop_last:
                    yield list(batch)
                batch = []
                batch_text_keys = set()
                active_cuis = deferred_cuis
                rng.shuffle(active_cuis)
                continue

            if not made_progress and deferred_cuis and not batch:
                # Fallback: preserve progress even when text-level constraints are too strict.
                fallback_batch = []
                for cui in deferred_cuis:
                    indices = groups[cui]
                    if not indices:
                        continue
                    fallback_batch.append(indices.popleft())
                    if indices:
                        next_active_cuis.append(cui)
                    if len(fallback_batch) == self.batch_size:
                        yield list(fallback_batch)
                        fallback_batch = []
                if fallback_batch and not self.drop_last:
                    yield list(fallback_batch)
                active_cuis = next_active_cuis
                rng.shuffle(active_cuis)
                continue

            active_cuis = next_active_cuis + deferred_cuis
            rng.shuffle(active_cuis)

        if batch and not self.drop_last:
            yield list(batch)

    def __iter__(self):
        yield from self._iter_batches()

    def __len__(self):
        if self._cached_num_batches is None:
            self._cached_num_batches = sum(1 for _ in self._iter_batches())
        return self._cached_num_batches


def build_no_duplicate_cui_batch_sampler(
    dataset,
    batch_size: int,
    drop_last: bool,
    valid_label_columns=None,
    generator=None,
    seed: int = 0,
    text_keys_per_example=None,
    deduplicate_by_text: bool = False,
):
    """Build the no-duplicate-CUI batch sampler."""
    sampler_seed = int(seed)
    if generator is not None:
        try:
            sampler_seed = int(generator.initial_seed())
        except Exception:
            sampler_seed = int(seed)

    effective_text_keys = text_keys_per_example
    if effective_text_keys is not None and len(effective_text_keys) != len(dataset):
        logger.warning(
            "Provided text_keys_per_example length (%d) does not match dataset length (%d); "
            "rebuilding text keys for the current dataset.",
            len(effective_text_keys),
            len(dataset),
        )
        effective_text_keys = None

    return NoDuplicateCuiBatchSampler(
        cuis=dataset["CUI"],
        text_keys_per_example=effective_text_keys if effective_text_keys is not None else build_example_text_keys(dataset),
        batch_size=batch_size,
        drop_last=drop_last,
        seed=sampler_seed,
        deduplicate_by_text=deduplicate_by_text,
    )


def build_example_text_keys(rows) -> list[set[str]]:
    """Build text keys used to group dense training examples."""
    column_names = rows.column_names if hasattr(rows, "column_names") else rows.columns
    mention_column = RAW_MENTION_COLUMN if RAW_MENTION_COLUMN in column_names else "mention_text"
    text_columns = [
        column_name
        for column_name in column_names
        if column_name in {mention_column, "concept_name"}
    ]

    text_keys_per_example = []
    for row_idx in range(len(rows)):
        row_text_keys = set()
        for column_name in text_columns:
            column_values = rows[column_name]
            text_value = (
                column_values.iloc[row_idx]
                if hasattr(column_values, "iloc")
                else column_values[row_idx]
            )
            if text_value is None or pd.isna(text_value):
                continue
            text_key = str(text_value)
            if text_key:
                row_text_keys.add(text_key)
        text_keys_per_example.append(row_text_keys)
    return text_keys_per_example

