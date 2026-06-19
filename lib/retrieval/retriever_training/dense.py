"""Dense retriever fine-tuning loop and callbacks."""

import logging
import shutil
import inspect
from functools import partial
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from transformers import TrainerCallback

from ..pipelines import evaluate_dense_retrieval
from .batching import build_example_text_keys, build_no_duplicate_cui_batch_sampler
from .data import (
    RAW_MENTION_COLUMN,
    build_dense_training_examples,
    build_dense_training_pairs,
)
from lib.utils.logging_utils import log_timed


logger = logging.getLogger(__name__)


def configure_training_precision(cfg: Dict) -> None:
    """Apply precision settings for dense retriever training."""
    use_fp16 = bool(cfg.get("USE_FP16", False))
    use_bf16 = bool(cfg.get("USE_BF16", False))
    allow_tf32 = bool(cfg.get("ALLOW_TF32", False))

    if use_fp16 and use_bf16:
        raise ValueError("USE_FP16 and USE_BF16 cannot both be enabled.")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    logger.info(
        "Configured training precision: use_fp16=%s, use_bf16=%s, allow_tf32=%s",
        use_fp16,
        use_bf16,
        allow_tf32,
    )


def get_context_special_tokens(context_cfg: Dict | None) -> list[str]:
    """Return special tokens used for contextualized retriever inputs."""
    context_cfg = context_cfg or {}
    context_format = str(context_cfg.get("FORMAT", "sep_token"))
    sep_token = str(context_cfg.get("SEP_TOKEN") or "[SEP]").strip()

    if context_format == "sep_token":
        return [sep_token] if sep_token else []
    if context_format == "explicit_markers":
        if str(context_cfg.get("MODE", "")) == "text_window":
            return ["[MENTION_START]", "[MENTION_END]"]
        return ["[MENTION]", "[CTX]"]
    return []


def register_context_special_tokens(model: SentenceTransformer, context_cfg: Dict | None) -> list[str]:
    """Register context special tokens in a sentence-transformer model."""
    special_tokens = [token for token in get_context_special_tokens(context_cfg) if token]
    if not special_tokens:
        return []

    tokenizer = getattr(model, "tokenizer", None)
    transformer_module = None
    auto_model = None
    if tokenizer is None:
        try:
            transformer_module = model._first_module()
        except Exception:
            transformer_module = None
        tokenizer = getattr(transformer_module, "tokenizer", None)
        auto_model = getattr(transformer_module, "auto_model", None)
    else:
        try:
            transformer_module = model._first_module()
        except Exception:
            transformer_module = None
        auto_model = getattr(transformer_module, "auto_model", None)

    if tokenizer is None:
        logger.warning("Could not register context special tokens because tokenizer is unavailable")
        return []

    tokens_to_add = []
    vocab = tokenizer.get_vocab()
    sep_token = str((context_cfg or {}).get("SEP_TOKEN") or "").strip()
    if sep_token and sep_token in special_tokens and tokenizer.sep_token != sep_token and sep_token not in vocab:
        tokenizer.add_special_tokens({"sep_token": sep_token})
        vocab = tokenizer.get_vocab()

    for token in special_tokens:
        if token == tokenizer.sep_token:
            continue
        if token not in vocab:
            tokens_to_add.append(token)

    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    if auto_model is not None:
        auto_model.resize_token_embeddings(len(tokenizer))

    logger.info("Registered context special tokens: %s", special_tokens)
    return special_tokens


class DenseRetrievalEvalCallback(TrainerCallback):
    """Evaluate dense retrieval quality during retriever fine-tuning."""
    def __init__(
        self,
        *,
        dev_df: pd.DataFrame,
        vocab_df: pd.DataFrame,
        cfg: Dict,
        best_model_dir,
        mention_column: str = "text",
    ):
        self.dev_df = dev_df
        self.vocab_df = vocab_df
        self.cfg = cfg
        self.best_model_dir = Path(best_model_dir)
        self.mention_column = str(mention_column)
        self.history_rows = []
        self.best_epoch = None
        self.best_metrics = None
        self.early_stopping_patience = max(int(self.cfg.get("EARLY_STOPPING_PATIENCE", 3)), 0)
        self.num_bad_epochs = 0
        self.stopped_early = False
        self.stop_epoch = None

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(round(float(state.epoch or 0)))
        epoch_row = {"epoch": epoch, "early_stopping_patience": self.early_stopping_patience}

        if not self.cfg.get("EVAL_EVERY_EPOCH", True):
            epoch_row["is_best"] = False
            epoch_row["num_bad_epochs"] = self.num_bad_epochs
            epoch_row["should_stop"] = False
            self.history_rows.append(epoch_row)
            return control

        _, dev_metrics = evaluate_dense_retrieval(
            data_df=self.dev_df,
            vocab_df=self.vocab_df,
            st_model=model,
            mention_column=self.mention_column,
            topk=self.cfg["DEV_TOPK"],
            query_batch_size=self.cfg["QUERY_BATCH_SIZE"],
            dense_vocab_batch_size=self.cfg["DENSE_VOCAB_BATCH_SIZE"],
            st_encode_batch_size=self.cfg["ST_ENCODE_BATCH_SIZE"],
            deduplicate_by_cui=self.cfg["DEDUPLICATE_BY_CUI"],
            resource_cache={},
        )
        epoch_row.update({f"dev_{metric_name}": float(metric_value) for metric_name, metric_value in dev_metrics.items()})

        selection_metric = str(self.cfg["SELECTION_METRIC"])
        current_score = float(dev_metrics[selection_metric])
        best_score = None if self.best_metrics is None else float(self.best_metrics[selection_metric])
        is_best = best_score is None or current_score > best_score
        epoch_row["is_best"] = bool(is_best)

        if is_best:
            self.best_epoch = epoch
            self.best_metrics = dev_metrics
            self.num_bad_epochs = 0
            if self.best_model_dir.exists():
                shutil.rmtree(self.best_model_dir)
            model.save(str(self.best_model_dir))
            logger.info(
                "Saved new best model at epoch=%d to %s with %s=%.6f",
                epoch,
                self.best_model_dir,
                selection_metric,
                current_score,
            )
        else:
            self.num_bad_epochs += 1
            logger.info(
                "Epoch=%d did not improve %s: current=%.6f best=%.6f",
                epoch,
                selection_metric,
                current_score,
                best_score,
            )
            if self.num_bad_epochs >= self.early_stopping_patience:
                control.should_training_stop = True
                self.stopped_early = True
                self.stop_epoch = epoch
                logger.info(
                    "Early stopping triggered at epoch=%d after %d consecutive non-improving epochs on %s",
                    epoch,
                    self.num_bad_epochs,
                    selection_metric,
                )

        epoch_row["num_bad_epochs"] = self.num_bad_epochs
        epoch_row["should_stop"] = bool(control.should_training_stop)
        self.history_rows.append(epoch_row)
        return control


def load_sentence_transformer_model(
    model_name: str,
    *,
    device: str,
    max_seq_length: int | None = None,
    context_cfg: Dict | None = None,
) -> SentenceTransformer:
    """Load a sentence-transformer model with project defaults."""
    model = SentenceTransformer(model_name, device=device)
    register_context_special_tokens(model, context_cfg)
    if max_seq_length is not None:
        model.max_seq_length = int(max_seq_length)
    logger.info(
        "Loaded SentenceTransformer model=%s on device=%s with max_seq_length=%s",
        model_name,
        device,
        max_seq_length,
    )
    return model


def build_train_dataset(train_examples_df: pd.DataFrame) -> HFDataset:
    """Build the dense retriever train dataset."""
    dataset = HFDataset.from_pandas(train_examples_df, preserve_index=False)
    logger.info(
        "Built Hugging Face training dataset with columns=%s and num_rows=%d",
        dataset.column_names,
        len(dataset),
    )
    return dataset


def build_eval_dataset(
    dev_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    *,
    mention_column: str = "text",
) -> HFDataset:
    """Build the dense retriever evaluation dataset."""
    dev_pairs_df = build_dense_training_pairs(
        entities_df=dev_df,
        vocab_df=vocab_df,
        mention_column=mention_column,
    )
    dataset = HFDataset.from_pandas(dev_pairs_df, preserve_index=False)
    logger.info(
        "Built Hugging Face eval dataset with columns=%s and num_rows=%d",
        dataset.column_names,
        len(dataset),
    )
    return dataset


def build_multiple_negatives_loss(model: SentenceTransformer):
    """Build the dense retriever contrastive loss."""
    return losses.MultipleNegativesRankingLoss(model=model)


def build_training_arguments(output_dir, cfg: Dict, text_keys_per_example=None):
    """Build sentence-transformers training arguments."""
    if cfg.get("CUI_AWARE_BATCHING", False):
        batch_sampler = (
            partial(
                build_no_duplicate_cui_batch_sampler,
                text_keys_per_example=text_keys_per_example,
                deduplicate_by_text=bool(cfg.get("BATCH_DEDUPLICATE_BY_TEXT", False)),
            )
            if text_keys_per_example is not None
            else partial(
                build_no_duplicate_cui_batch_sampler,
                deduplicate_by_text=bool(cfg.get("BATCH_DEDUPLICATE_BY_TEXT", False)),
            )
        )
    else:
        batch_sampler = BatchSamplers.BATCH_SAMPLER
    logging_steps = int(cfg.get("TRAIN_LOGGING_STEPS", 50))
    dev_loss_eval_steps = int(cfg.get("DEV_LOSS_EVAL_STEPS", 0))
    logging_strategy = "steps" if logging_steps > 0 else "epoch"
    eval_strategy = "steps" if dev_loss_eval_steps > 0 else "epoch"
    training_kwargs = dict(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        eval_strategy=eval_strategy,
        eval_steps=dev_loss_eval_steps if dev_loss_eval_steps > 0 else None,
        save_strategy="epoch",
        logging_strategy=logging_strategy,
        logging_steps=logging_steps if logging_steps > 0 else 500,
        per_device_train_batch_size=int(cfg["TRAIN_BATCH_SIZE"]),
        per_device_eval_batch_size=int(cfg.get("EVAL_BATCH_SIZE", cfg["TRAIN_BATCH_SIZE"])),
        gradient_accumulation_steps=int(cfg.get("GRAD_ACCUMULATION_STEPS", 1)),
        num_train_epochs=float(cfg["EPOCHS"]),
        learning_rate=float(cfg["LEARNING_RATE"]),
        weight_decay=float(cfg["WEIGHT_DECAY"]),
        warmup_ratio=float(cfg["WARMUP_RATIO"]),
        fp16=bool(cfg.get("USE_FP16", False)),
        bf16=bool(cfg.get("USE_BF16", False)),
        seed=int(cfg["SEED"]),
        remove_unused_columns=False,
        report_to=[],
        batch_sampler=batch_sampler,
        dataloader_drop_last=False,
    )
    signature = inspect.signature(SentenceTransformerTrainingArguments.__init__)
    if "tf32" in signature.parameters:
        training_kwargs["tf32"] = bool(cfg.get("ALLOW_TF32", False))
    if "save_safetensors" in signature.parameters:
        training_kwargs["save_safetensors"] = True
    return SentenceTransformerTrainingArguments(**training_kwargs)


def _finalize_training_result(trainer, eval_callback, cfg: Dict):
    history_df = pd.DataFrame(trainer.state.log_history)
    if not history_df.empty and "epoch" in history_df.columns:
        history_df["epoch_int"] = history_df["epoch"].round().astype("Int64")
    callback_history_df = pd.DataFrame(eval_callback.history_rows)
    if not callback_history_df.empty:
        callback_history_df["epoch_int"] = callback_history_df["epoch"].astype("Int64")
    if not history_df.empty and not callback_history_df.empty:
        history_df = history_df.merge(
            callback_history_df.drop(columns=["epoch"], errors="ignore"),
            on="epoch_int",
            how="left",
        )
    elif history_df.empty:
        history_df = callback_history_df
    best_epoch = eval_callback.best_epoch if eval_callback.best_epoch is not None else int(cfg["EPOCHS"])
    best_metrics = eval_callback.best_metrics if eval_callback.best_metrics is not None else {}
    return history_df, int(best_epoch), best_metrics


def train_dense_retriever(
    *,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    model_name: str,
    output_dir,
    cfg: Dict,
    mention_column: str = "text",
    raw_mention_column: str | None = None,
):
    """Train a dense retriever model."""
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)

    configure_training_precision(cfg)

    model = load_sentence_transformer_model(
        model_name=model_name,
        device=cfg["DEVICE"],
        max_seq_length=cfg.get("MAX_SEQ_LENGTH"),
        context_cfg=cfg.get("CONTEXT"),
    )

    train_examples_df = build_dense_training_examples(
        entities_df=train_df,
        vocab_df=vocab_df,
        mention_column=mention_column,
        raw_mention_column=raw_mention_column,
        st_model=model,
        model_id=model_name,
        query_batch_size=cfg["QUERY_BATCH_SIZE"],
        dense_vocab_batch_size=cfg["DENSE_VOCAB_BATCH_SIZE"],
        st_encode_batch_size=cfg["ST_ENCODE_BATCH_SIZE"],
        num_hard_negatives=cfg.get("NUM_HARD_NEGATIVES", 0),
        hard_negative_deduplicate_by_cui=cfg.get("HARD_NEGATIVE_DEDUPLICATE_BY_CUI", True),
        hard_negative_skip_topk=cfg.get("HARD_NEGATIVE_SKIP_TOPK", 0),
        hard_negative_cache_dir=cfg.get("HARD_NEGATIVE_CACHE_DIR"),
    )
    train_dataset = build_train_dataset(
        train_examples_df=train_examples_df.drop(columns=[RAW_MENTION_COLUMN], errors="ignore")
    )
    eval_dataset = build_eval_dataset(
        dev_df=dev_df,
        vocab_df=vocab_df,
        mention_column=mention_column,
    )
    train_loss = build_multiple_negatives_loss(model=model)
    training_args = build_training_arguments(
        output_dir=checkpoints_dir,
        cfg=cfg,
        text_keys_per_example=build_example_text_keys(train_examples_df),
    )
    eval_callback = DenseRetrievalEvalCallback(
        dev_df=dev_df,
        vocab_df=vocab_df,
        cfg=cfg,
        best_model_dir=best_model_dir,
        mention_column=mention_column,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        callbacks=[eval_callback],
    )

    logger.info(
        "Starting dense retriever training: mention_column=%s, raw_mention_column=%s, num_train_rows=%d, num_hard_negative_columns=%d, batch_size=%d, cui_aware_batching=%s, batch_deduplicate_by_text=%s",
        mention_column,
        raw_mention_column,
        len(train_examples_df),
        len([column for column in train_examples_df.columns if column.startswith("hard_negative_")]),
        int(cfg["TRAIN_BATCH_SIZE"]),
        bool(cfg.get("CUI_AWARE_BATCHING", False)),
        bool(cfg.get("BATCH_DEDUPLICATE_BY_TEXT", False)),
    )

    with log_timed(logger, f"Dense retriever training for {cfg['EPOCHS']} epochs"):
        trainer.train()

    if not best_model_dir.exists():
        logger.info("No best model was saved during callbacks; saving final model to %s", best_model_dir)
        model.save(str(best_model_dir))

    history_df, best_epoch, best_metrics = _finalize_training_result(
        trainer=trainer,
        eval_callback=eval_callback,
        cfg=cfg,
    )

    return {
        "train_pairs_df": train_examples_df,
        "history_df": history_df,
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "stopped_early": bool(eval_callback.stopped_early),
        "stop_epoch": None if eval_callback.stop_epoch is None else int(eval_callback.stop_epoch),
        "best_checkpoint_dir": str(best_model_dir),
        "best_model_dir": str(best_model_dir),
    }
