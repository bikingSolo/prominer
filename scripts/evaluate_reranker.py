"""Evaluate a cross-encoder reranker on top of a dense retriever cache."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.data.vocab_enrichment import (
    enrich_vocab_with_oov_train_dev_terms,
    filter_vocab_for_dataset_language,
    prepare_experiment_vocab,
)
from lib.retrieval.reranking.candidate_cache import (
    build_retriever_candidate_cache,
    load_retriever_candidate_cache,
    save_retriever_candidate_cache,
)
from lib.retrieval.reranking.candidate_context import build_candidate_context_cache
from lib.retrieval.reranking.candidate_context_cache import (
    build_candidate_text_map,
    load_candidate_context_cache,
    save_candidate_context_cache,
)
from lib.retrieval.reranking.dictionary_pretrain.artifacts import build_dictionary_pretrain_cache_metadata
from lib.retrieval.reranking.dictionary_pretrain.fingerprints import fingerprint_dictionary_pretrain_dataframe
from lib.retrieval.reranking.inference import rerank_from_candidate_cache
from lib.retrieval.reranking.io import load_cross_encoder_model_with_config, save_json
from lib.retrieval.tuning import evaluate_dev_predictions

from eval_common import (
    add_common_data_args,
    add_common_runtime_args,
    build_metrics_table,
    configure_cli_logging,
    dataset_lang_value,
    load_eval_data,
    prepare_eval_vocabs,
    print_metrics_table,
    resolve_device,
    set_seed,
    write_outputs,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retriever-model",
        default="bikingSolo/prominer-ru-retriever",
        help="SentenceTransformer checkpoint path or HF model id.",
    )
    parser.add_argument(
        "--reranker-model",
        default="bikingSolo/prominer-ru-reranker",
        help="CrossEncoder checkpoint path or HF model id.",
    )
    add_common_data_args(parser)
    add_common_runtime_args(parser)
    parser.set_defaults(query_batch_size=262_144, st_encode_batch_size=1024)
    parser.add_argument("--dev-candidate-pool-size", type=int, default=20)
    parser.add_argument("--test-candidate-pool-size", type=int, default=20)
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--disable-candidate-context", action="store_true")
    parser.add_argument("--candidate-context-cache", default=None)
    parser.add_argument("--candidate-context-stem", default="candidate_context_short")
    parser.add_argument("--candidate-max-aliases", type=int, default=6)
    parser.add_argument("--candidate-num-workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_retriever(model_name_or_path: str, *, device: str, local_files_only: bool, trust_remote_code: bool):
    return SentenceTransformer(
        model_name_or_path,
        device=device,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )


def load_reranker(args):
    model_cfg = {
        "TRUST_REMOTE_CODE": bool(args.trust_remote_code),
        "LOCAL_FILES_ONLY": bool(args.local_files_only),
        "TORCH_DTYPE": args.torch_dtype,
    }
    return load_cross_encoder_model_with_config(args.reranker_model, device=args.device, cfg=model_cfg)


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def infer_cache_topk(cache: dict) -> int:
    max_topk = 0
    for payload in cache.values():
        scores = payload.get("candidate_scores")
        if scores is not None and hasattr(scores, "shape") and len(scores.shape) == 2:
            max_topk = max(max_topk, int(scores.shape[1]))
    return max_topk


def trim_cache(cache: dict, topk: int) -> dict:
    trimmed = {}
    for entity_type, payload in cache.items():
        row = dict(payload)
        if payload.get("candidate_scores") is not None:
            row["candidate_scores"] = payload["candidate_scores"][:, :topk].copy()
        if payload.get("candidate_indices") is not None:
            row["candidate_indices"] = payload["candidate_indices"][:, :topk].copy()
        trimmed[entity_type] = row
    return trimmed


def fingerprint_queries(entities_df: pd.DataFrame, *, mention_column: str) -> str:
    columns = [mention_column, "document_id", "spans", "entity_type"]
    if "UMLS_CUI" in entities_df.columns:
        columns.append("UMLS_CUI")
    return fingerprint_dictionary_pretrain_dataframe(entities_df, columns=columns)


def expected_retriever_cache_metadata(args, vocab_df, data_df, *, split_name: str, topk: int, stem: str) -> dict:
    return build_dictionary_pretrain_cache_metadata(
        vocab_df=vocab_df,
        cfg={
            "retriever_model_name_or_path": str(args.retriever_model),
            "mention_column": str(args.mention_column),
            "topk": int(topk),
            "deduplicate_by_cui": not args.no_deduplicate_by_cui,
            "queries_fingerprint": fingerprint_queries(data_df, mention_column=args.mention_column),
            "cache_stem": str(stem),
            "split": str(split_name),
        },
    )


def can_reuse_retriever_cache(metadata_path: Path, expected_metadata: dict, requested_topk: int) -> bool:
    existing_metadata = read_json(metadata_path)
    if existing_metadata is None:
        return False
    existing_cfg = dict(existing_metadata.get("config", {}))
    expected_cfg = dict(expected_metadata.get("config", {}))
    existing_topk = int(existing_cfg.pop("topk", 0) or 0)
    expected_topk = int(expected_cfg.pop("topk", 0) or 0)
    return (
        existing_metadata.get("vocab_fingerprint") == expected_metadata.get("vocab_fingerprint")
        and existing_cfg == expected_cfg
        and existing_topk >= int(requested_topk)
        and existing_topk >= expected_topk
    )


def can_reuse_candidate_context(metadata_path: Path, expected_metadata: dict) -> bool:
    existing_metadata = read_json(metadata_path)
    if existing_metadata is None:
        return False
    return (
        existing_metadata.get("vocab_fingerprint") == expected_metadata.get("vocab_fingerprint")
        and existing_metadata.get("config") == expected_metadata.get("config")
    )


def prepare_retriever_cache(args, split_name: str, data_df, vocab_df, retriever_model, *, topk: int, cache_dir: Path):
    stem = f"{split_name}_retriever_cache"
    pickle_path = cache_dir / f"{stem}.pkl"
    metadata_path = cache_dir / f"{stem}_metadata.json"
    expected_metadata = expected_retriever_cache_metadata(
        args,
        vocab_df,
        data_df,
        split_name=split_name,
        topk=topk,
        stem=stem,
    )
    use_cache = (
        not args.force_rebuild_cache
        and pickle_path.exists()
        and can_reuse_retriever_cache(metadata_path, expected_metadata, topk)
    )
    if use_cache:
        LOGGER.info("Loading %s retriever cache from %s", split_name, pickle_path)
        cache = load_retriever_candidate_cache(pickle_path)
        cached_topk = infer_cache_topk(cache)
        if cached_topk > topk:
            cache = trim_cache(cache, topk)
        return cache

    LOGGER.info("Building %s retriever cache", split_name)
    cache = build_retriever_candidate_cache(
        entities_df=data_df,
        vocab_df=vocab_df,
        retriever_model=retriever_model,
        mention_column=args.mention_column,
        topk=topk,
        query_batch_size=args.query_batch_size,
        dense_vocab_batch_size=args.dense_vocab_batch_size,
        st_encode_batch_size=args.st_encode_batch_size,
        deduplicate_by_cui=not args.no_deduplicate_by_cui,
    )
    save_retriever_candidate_cache(cache, cache_dir, stem=stem)
    save_json(expected_metadata, metadata_path)
    return cache


def candidate_context_languages(dataset: str) -> list[str]:
    if dataset == "en":
        return ["ENG", "RUS"]
    return ["RUS", "ENG"]


def candidate_context_allowed_languages(dataset: str) -> list[str]:
    if dataset == "en":
        return ["ENG"]
    return []


def prepare_candidate_text_map(args, train_df, dev_df, raw_vocab_df, cache_dir: Path):
    if args.disable_candidate_context:
        return {}
    if args.candidate_context_cache:
        candidate_context_df, _ = load_candidate_context_cache(args.candidate_context_cache)
        return build_candidate_text_map(candidate_context_df)

    stem = args.candidate_context_stem
    cache_path = cache_dir / f"{stem}.parquet"
    metadata_path = cache_dir / f"{stem}_metadata.json"
    enrichment_entities_df = pd.concat([train_df, dev_df], ignore_index=True)
    context_vocab = raw_vocab_df.copy()
    context_text_column = "raw_text" if "raw_text" in enrichment_entities_df.columns else "text"
    if args.dataset == "en":
        context_vocab = filter_vocab_for_dataset_language(context_vocab, args.dataset)
    if not args.no_default_oov_enrichment:
        context_vocab = enrich_vocab_with_oov_train_dev_terms(
            context_vocab,
            enrichment_entities_df,
            text_column=context_text_column,
            lang_value=dataset_lang_value(args.dataset),
        )
    context_vocab = prepare_experiment_vocab(
        context_vocab,
        enrichment_entities_df,
        {"ENRICH_VOCABULARY": bool(args.enrich_vocab)},
        text_column=context_text_column,
        lang_value=dataset_lang_value(args.dataset),
    )
    context_vocab = filter_vocab_for_dataset_language(context_vocab, args.dataset)
    context_cfg = {
        "alias_length_threshold": None,
        "max_aliases": int(args.candidate_max_aliases),
        "group_limits": {
            "abbreviations": 1,
            "short_names": 2,
            "multi_word": 2,
            "long_variants": 1,
        },
        "preferred_languages": candidate_context_languages(args.dataset),
        "allowed_languages": candidate_context_allowed_languages(args.dataset),
        "cache_stem": stem,
    }
    expected_metadata = build_dictionary_pretrain_cache_metadata(vocab_df=context_vocab, cfg=context_cfg)
    if cache_path.exists() and can_reuse_candidate_context(metadata_path, expected_metadata) and not args.force_rebuild_cache:
        candidate_context_df, _ = load_candidate_context_cache(cache_path, metadata_path)
    else:
        candidate_context_df, _ = build_candidate_context_cache(
            context_vocab,
            alias_length_threshold=None,
            max_aliases=args.candidate_max_aliases,
            group_limits=context_cfg["group_limits"],
            preferred_languages=context_cfg["preferred_languages"],
            allowed_languages=context_cfg["allowed_languages"],
            num_workers=args.candidate_num_workers,
        )
        save_candidate_context_cache(candidate_context_df, expected_metadata, cache_dir, stem=stem)
    return build_candidate_text_map(candidate_context_df)


def evaluate_split(args, split_name, data_df, vocab_df, retriever_cache, reranker_model, candidate_text_map):
    return_topk = args.dev_topk if split_name == "dev" else args.test_topk
    predictions_df = rerank_from_candidate_cache(
        data_df=data_df,
        retriever_cache=retriever_cache,
        cross_encoder_model=reranker_model,
        mention_column=args.mention_column,
        candidate_text_map=candidate_text_map,
        batch_size=args.rerank_batch_size,
        topk=return_topk,
    )
    metrics = evaluate_dev_predictions(predictions_df=predictions_df, data_df=data_df)
    return predictions_df, {"split": split_name, **metrics}


def main() -> None:
    args = parse_args()
    configure_cli_logging(args.verbose)
    set_seed(args.seed)
    args.device = resolve_device(args.device)

    train_df, dev_df, test_df, vocab_df = load_eval_data(args, preserve_raw_text=True)
    raw_vocab_df = pd.read_parquet(args.vocab_path)
    dev_vocab, test_vocab, _ = prepare_eval_vocabs(args, train_df, dev_df, vocab_df)
    if args.dataset in {"en", "bilingual"}:
        # The cross-encoder notebook reuses the dev/training vocabulary for
        # EN and bilingual test cache construction.
        test_vocab = dev_vocab.copy()

    output_dir = Path(args.output_dir) / args.dataset / "reranker"
    cache_dir = output_dir / "retriever_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    candidate_text_map = prepare_candidate_text_map(args, train_df, dev_df, raw_vocab_df, output_dir / "candidate_context")
    retriever_model = load_retriever(
        args.retriever_model,
        device=args.device,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    reranker_model = load_reranker(args)

    dev_cache = prepare_retriever_cache(
        args,
        "dev",
        dev_df,
        dev_vocab,
        retriever_model,
        topk=args.dev_candidate_pool_size,
        cache_dir=cache_dir,
    )
    test_cache = prepare_retriever_cache(
        args,
        "test",
        test_df,
        test_vocab,
        retriever_model,
        topk=args.test_candidate_pool_size,
        cache_dir=cache_dir,
    )

    dev_predictions, dev_metrics = evaluate_split(args, "dev", dev_df, dev_vocab, dev_cache, reranker_model, candidate_text_map)
    test_predictions, test_metrics = evaluate_split(args, "test", test_df, test_vocab, test_cache, reranker_model, candidate_text_map)
    metrics_table = build_metrics_table([dev_metrics, test_metrics])
    paths = write_outputs(
        output_dir=output_dir,
        metrics_table=metrics_table,
        predictions={"dev": dev_predictions, "test": test_predictions},
        save_predictions=args.save_predictions,
    )

    print_metrics_table(metrics_table)
    print(f"\nSaved metrics: {paths['metrics']}")


if __name__ == "__main__":
    main()
