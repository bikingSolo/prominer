"""Evaluate a dense retriever checkpoint on BioNNE-L dev and test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.retrieval.pipelines.dense_only import make_dense_predictions
from lib.retrieval.tuning import evaluate_dev_predictions

from eval_common import (
    add_common_data_args,
    add_common_runtime_args,
    build_metrics_table,
    configure_cli_logging,
    load_eval_data,
    prepare_eval_vocabs,
    print_metrics_table,
    resolve_device,
    set_seed,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retriever-model",
        default="bikingSolo/prominer-ru-retriever",
        help="SentenceTransformer checkpoint path or Hugging Face model id.",
    )
    add_common_data_args(parser)
    add_common_runtime_args(parser)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_retriever(model_name_or_path: str, *, device: str, local_files_only: bool, trust_remote_code: bool):
    kwargs = {
        "device": device,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    return SentenceTransformer(model_name_or_path, **kwargs)


def evaluate_split(args, split_name: str, data_df, vocab_df, model):
    topk = args.dev_topk if split_name == "dev" else args.test_topk
    predictions_df = make_dense_predictions(
        data_df=data_df,
        vocab_df=vocab_df,
        st_model=model,
        mention_column=args.mention_column,
        topk=topk,
        query_batch_size=args.query_batch_size,
        dense_vocab_batch_size=args.dense_vocab_batch_size,
        st_encode_batch_size=args.st_encode_batch_size,
        deduplicate_by_cui=not args.no_deduplicate_by_cui,
        resource_cache={},
    )
    metrics = evaluate_dev_predictions(predictions_df=predictions_df, data_df=data_df)
    return predictions_df, {"split": split_name, **metrics}


def main() -> None:
    args = parse_args()
    configure_cli_logging(args.verbose)
    set_seed(args.seed)
    args.device = resolve_device(args.device)

    train_df, dev_df, test_df, vocab_df = load_eval_data(args)
    dev_vocab, test_vocab, _ = prepare_eval_vocabs(args, train_df, dev_df, vocab_df)
    model = load_retriever(
        args.retriever_model,
        device=args.device,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )

    dev_predictions, dev_metrics = evaluate_split(args, "dev", dev_df, dev_vocab, model)
    test_predictions, test_metrics = evaluate_split(args, "test", test_df, test_vocab, model)

    metrics_table = build_metrics_table([dev_metrics, test_metrics])
    output_dir = Path(args.output_dir) / args.dataset / "retriever"
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
