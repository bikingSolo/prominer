"""Shared CLI helpers for checkpoint evaluation scripts."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.data.text_preprocessing import preprocess_text
from lib.data.vocab_enrichment import (
    enrich_vocab_with_oov_train_dev_terms,
    filter_vocab_for_dataset_language,
    prepare_experiment_vocab,
)


LOGGER = logging.getLogger(__name__)


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=["ru", "en", "bilingual"], default="ru")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--vocab-path", default="data/vocabular/bionnel_vocab_bilingual.parquet")
    parser.add_argument("--dev-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--enrich-vocab", action="store_true")
    parser.add_argument(
        "--no-test-enrich-vocab",
        action="store_true",
        help="Disable notebook-style TEST_ENRICH_VOCABULARY for test inference.",
    )
    parser.add_argument(
        "--enrichment-datasets",
        default=None,
        help="Comma-separated datasets used for train/dev vocabulary enrichment. Defaults match the notebooks.",
    )
    parser.add_argument("--no-default-oov-enrichment", action="store_true")
    parser.add_argument("--output-dir", default="artifacts_checkpoint_eval")
    parser.add_argument("--save-predictions", action="store_true")


def add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Device for model inference. Defaults to cuda if available, else cpu.")
    parser.add_argument("--mention-column", default="text")
    parser.add_argument("--dev-topk", type=int, default=20)
    parser.add_argument("--test-topk", type=int, default=5)
    parser.add_argument("--query-batch-size", type=int, default=131_072)
    parser.add_argument("--dense-vocab-batch-size", type=int, default=16_384)
    parser.add_argument("--st-encode-batch-size", type=int, default=8_192)
    parser.add_argument("--no-deduplicate-by-cui", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", default=None)


def resolve_device(device: str | None) -> str:
    if device:
        return str(device)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        LOGGER.debug("Torch is unavailable; skipped torch seed setup.", exc_info=True)


def configure_cli_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def resolve_split_path(args, split: str) -> Path:
    explicit = getattr(args, f"{split}_path", None)
    if explicit:
        return Path(explicit)
    dataset = args.dataset
    data_root = Path(args.data_root)
    if split == "test":
        tsv_path = data_root / "tsv" / dataset / f"bionnel_{dataset}_test.tsv"
        if tsv_path.exists():
            return tsv_path
    return data_root / "parquet" / dataset / f"bionnel_{dataset}_{split}.parquet"


def read_entities(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t")


def preprocess_entities(df: pd.DataFrame, *, preserve_raw_text: bool = False) -> pd.DataFrame:
    result_df = df.copy()
    if preserve_raw_text and "raw_text" not in result_df.columns:
        result_df["raw_text"] = result_df["text"].astype(str)
    result_df["text"] = result_df["text"].map(preprocess_text)
    return result_df


def preprocess_vocab(vocab_df: pd.DataFrame) -> pd.DataFrame:
    result_df = vocab_df.copy()
    result_df["concept_name"] = result_df["concept_name"].map(preprocess_text)
    return result_df


def dataset_lang_value(dataset: str) -> str | None:
    if dataset == "en":
        return "EN"
    return None


def load_eval_data(args, *, preserve_raw_text: bool = False):
    train_path = resolve_split_path(args, "train")
    dev_path = resolve_split_path(args, "dev")
    test_path = resolve_split_path(args, "test")
    vocab_path = Path(args.vocab_path)

    train_df = read_entities(train_path)
    dev_df = read_entities(dev_path)
    test_df = read_entities(test_path)
    vocab_df = pd.read_parquet(vocab_path)

    if not args.no_preprocess:
        train_df = preprocess_entities(train_df, preserve_raw_text=preserve_raw_text)
        dev_df = preprocess_entities(dev_df, preserve_raw_text=preserve_raw_text)
        test_df = preprocess_entities(test_df, preserve_raw_text=preserve_raw_text)
        vocab_df = preprocess_vocab(vocab_df)

    return train_df, dev_df, test_df, vocab_df


def default_enrichment_datasets(dataset: str) -> list[str]:
    if dataset in {"ru", "bilingual"}:
        return ["ru", "en", "bilingual"]
    return ["en"]


def resolve_enrichment_datasets(args) -> list[str]:
    if args.enrichment_datasets:
        return [part.strip().lower() for part in args.enrichment_datasets.split(",") if part.strip()]
    return default_enrichment_datasets(args.dataset)


def load_enrichment_entities(args, current_train_df: pd.DataFrame, current_dev_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for dataset_name in resolve_enrichment_datasets(args):
        if dataset_name == args.dataset:
            frames.extend([current_train_df, current_dev_df])
            continue

        data_root = Path(args.data_root)
        train_df = read_entities(data_root / "parquet" / dataset_name / f"bionnel_{dataset_name}_train.parquet")
        dev_df = read_entities(data_root / "parquet" / dataset_name / f"bionnel_{dataset_name}_dev.parquet")

        if not args.no_preprocess:
            preserve_raw_text = "raw_text" in current_train_df.columns or "raw_text" in current_dev_df.columns
            train_df = preprocess_entities(train_df, preserve_raw_text=preserve_raw_text)
            dev_df = preprocess_entities(dev_df, preserve_raw_text=preserve_raw_text)
        frames.extend([train_df, dev_df])
    return pd.concat(frames, ignore_index=True)


def prepare_eval_vocabs(args, train_df: pd.DataFrame, dev_df: pd.DataFrame, vocab_df: pd.DataFrame):
    enrichment_entities_df = load_enrichment_entities(args, train_df, dev_df)
    lang_value = dataset_lang_value(args.dataset)

    dev_vocab = vocab_df.copy()
    if not args.no_default_oov_enrichment:
        dev_vocab = enrich_vocab_with_oov_train_dev_terms(
            dev_vocab,
            enrichment_entities_df,
            lang_value=lang_value,
        )
    dev_vocab = prepare_experiment_vocab(
        dev_vocab,
        enrichment_entities_df,
        {"ENRICH_VOCABULARY": bool(args.enrich_vocab)},
        lang_value=lang_value,
    )
    dev_vocab = filter_vocab_for_dataset_language(dev_vocab, args.dataset)

    test_vocab = vocab_df.copy()
    if not args.no_default_oov_enrichment:
        test_vocab = enrich_vocab_with_oov_train_dev_terms(
            test_vocab,
            enrichment_entities_df,
            lang_value=lang_value,
        )
    test_vocab = prepare_experiment_vocab(
        test_vocab,
        enrichment_entities_df,
        {"ENRICH_VOCABULARY": not bool(args.no_test_enrich_vocab)},
        lang_value=lang_value,
    )
    test_vocab = filter_vocab_for_dataset_language(test_vocab, args.dataset)
    return dev_vocab.reset_index(drop=True), test_vocab.reset_index(drop=True), enrichment_entities_df


def build_metrics_table(rows: Iterable[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    metric_columns = [column for column in ["Acc@1", "Acc@5", "Acc@10", "Acc@20", "MRR"] if column in table.columns]
    return table[["split", *metric_columns]]


def write_outputs(
    *,
    output_dir: Path,
    metrics_table: pd.DataFrame,
    predictions: dict[str, pd.DataFrame] | None = None,
    save_predictions: bool = False,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    metrics_path = output_dir / "metrics_summary.tsv"
    metrics_table.to_csv(metrics_path, sep="\t", index=False)
    paths["metrics"] = str(metrics_path)

    json_path = output_dir / "metrics_summary.json"
    json_path.write_text(
        json.dumps(metrics_table.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["metrics_json"] = str(json_path)

    if save_predictions and predictions:
        for split_name, pred_df in predictions.items():
            pred_path = output_dir / f"{split_name}_predictions.tsv"
            pred_df.to_csv(pred_path, sep="\t", index=False)
            paths[f"{split_name}_predictions"] = str(pred_path)
    return paths


def print_metrics_table(metrics_table: pd.DataFrame) -> None:
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(metrics_table.to_string(index=False))
