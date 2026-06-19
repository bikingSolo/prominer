"""Train and evaluate a cross-encoder for dictionary pretraining."""

import inspect
import logging
import shutil
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from transformers import TrainerCallback

from lib.utils.logging_utils import log_timed


logger = logging.getLogger(__name__)


def build_cross_encoder_pretrain_dataset(pairwise_df: pd.DataFrame) -> HFDataset:
    """Convert pairwise rows into a Hugging Face dataset."""
    dataset = HFDataset.from_pandas(pairwise_df[["query", "candidate_text", "label"]], preserve_index=False)
    logger.info("Built dictionary pretrain dataset with num_rows=%d", len(dataset))
    return dataset


def compute_dictionary_pretrain_ranking_metrics(
    pairwise_df: pd.DataFrame,
    cross_encoder_model: CrossEncoder,
    *,
    split_name: str = "dev",
    batch_size: int = 64,
) -> Dict[str, float]:
    """Evaluate cross-encoder ranking quality on pairwise candidates."""
    split_df = pairwise_df[pairwise_df["split"].astype(str) == str(split_name)].copy()
    if split_df.empty:
        return {}

    pair_inputs = list(zip(split_df["query"].astype(str).tolist(), split_df["candidate_text"].astype(str).tolist()))
    scores = cross_encoder_model.predict(
        pair_inputs,
        batch_size=int(batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    split_df["score"] = scores

    acc_at_1_hits = []
    acc_at_5_hits = []
    reciprocal_ranks = []
    for _, query_df in split_df.groupby("query_id", sort=False):
        query_df = query_df.sort_values(
            ["score", "retriever_score", "candidate_rank"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        labels = query_df["label"].astype(float).tolist()
        acc_at_1_hits.append(1.0 if labels and labels[0] > 0 else 0.0)
        acc_at_5_hits.append(1.0 if any(label > 0 for label in labels[:5]) else 0.0)

        rr = 0.0
        for rank, label in enumerate(labels, start=1):
            if label > 0:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return {
        "Acc@1": float(sum(acc_at_1_hits) / len(acc_at_1_hits)) if acc_at_1_hits else 0.0,
        "Acc@5": float(sum(acc_at_5_hits) / len(acc_at_5_hits)) if acc_at_5_hits else 0.0,
        "MRR": float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "NumQueries": int(len(acc_at_1_hits)),
    }


class DictionaryPretrainEvalCallback(TrainerCallback):
    """Evaluate and save the best dictionary-pretrained cross-encoder each epoch."""

    def __init__(
        self,
        *,
        dev_pairwise_df: pd.DataFrame,
        cfg: Dict,
        best_model_dir,
    ):
        self.dev_pairwise_df = dev_pairwise_df.copy()
        self.cfg = cfg
        self.best_model_dir = Path(best_model_dir)
        self.history_rows = []
        self.best_epoch = None
        self.best_metrics = None

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.dev_pairwise_df.empty:
            return control

        epoch = int(round(float(state.epoch or 0)))
        epoch_row = {"epoch": epoch}
        dev_metrics = compute_dictionary_pretrain_ranking_metrics(
            self.dev_pairwise_df,
            model,
            split_name="dev",
            batch_size=int(self.cfg.get("RERANK_BATCH_SIZE", 64)),
        )
        epoch_row.update({f"dev_{metric_name}": float(metric_value) for metric_name, metric_value in dev_metrics.items()})

        selection_metric = str(self.cfg.get("SELECTION_METRIC", "Acc@1"))
        current_score = float(dev_metrics[selection_metric])
        best_score = None if self.best_metrics is None else float(self.best_metrics[selection_metric])
        is_best = best_score is None or current_score > best_score
        epoch_row["is_best"] = bool(is_best)

        if is_best:
            self.best_epoch = epoch
            self.best_metrics = dev_metrics
            if self.best_model_dir.exists():
                shutil.rmtree(self.best_model_dir)
            model.save(str(self.best_model_dir), safe_serialization=True)
            logger.info(
                "Saved new best dictionary pretrain model at epoch=%d to %s with %s=%.6f",
                epoch,
                self.best_model_dir,
                selection_metric,
                current_score,
            )

        self.history_rows.append(epoch_row)
        return control


def build_dictionary_pretrain_training_arguments(output_dir, cfg: Dict):
    """Build sentence-transformers training arguments for dictionary pretraining."""
    training_kwargs = dict(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=bool(cfg.get("DO_EVAL", True)),
        eval_strategy="epoch" if bool(cfg.get("DO_EVAL", True)) else "no",
        save_strategy="epoch",
        logging_strategy="steps" if int(cfg.get("TRAIN_LOGGING_STEPS", 50)) > 0 else "epoch",
        logging_steps=int(cfg.get("TRAIN_LOGGING_STEPS", 50)) if int(cfg.get("TRAIN_LOGGING_STEPS", 50)) > 0 else 500,
        per_device_train_batch_size=int(cfg["TRAIN_BATCH_SIZE"]),
        per_device_eval_batch_size=int(cfg.get("EVAL_BATCH_SIZE", cfg["TRAIN_BATCH_SIZE"])),
        gradient_accumulation_steps=int(cfg.get("GRAD_ACCUMULATION_STEPS", 1)),
        num_train_epochs=float(cfg["EPOCHS"]),
        learning_rate=float(cfg["LEARNING_RATE"]),
        weight_decay=float(cfg["WEIGHT_DECAY"]),
        warmup_ratio=float(cfg["WARMUP_RATIO"]),
        seed=int(cfg["SEED"]),
        remove_unused_columns=False,
        report_to=[],
        dataloader_drop_last=False,
    )
    signature = inspect.signature(CrossEncoderTrainingArguments.__init__)
    if "save_safetensors" in signature.parameters:
        training_kwargs["save_safetensors"] = True
    if bool(cfg.get("DO_EVAL", True)):
        if "load_best_model_at_end" in signature.parameters:
            training_kwargs["load_best_model_at_end"] = False
    if "save_total_limit" in signature.parameters:
        training_kwargs["save_total_limit"] = 2
    return CrossEncoderTrainingArguments(**training_kwargs)


def _finalize_dictionary_pretrain_result(trainer, eval_callback):
    """Merge trainer logs with callback metrics and best-model metadata."""
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

    best_epoch = eval_callback.best_epoch
    best_metrics = eval_callback.best_metrics or {}
    return history_df, best_epoch, best_metrics


def train_dictionary_pretrain_cross_encoder(
    *,
    pairwise_df: pd.DataFrame,
    cross_encoder_model_name: str,
    output_dir,
    cfg: Dict,
) -> Dict[str, object]:
    """Train a cross-encoder on dictionary-derived pairwise examples."""
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    best_model_dir = output_dir / "best_model"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if pairwise_df.empty:
        raise ValueError("pairwise_df must not be empty.")

    train_pairs_df = pairwise_df[pairwise_df["split"].astype(str) == "train"].reset_index(drop=True)
    eval_pairs_df = pairwise_df[pairwise_df["split"].astype(str) == "dev"].reset_index(drop=True)
    if train_pairs_df.empty:
        raise ValueError("No train rows available for dictionary pretraining.")

    do_eval = not eval_pairs_df.empty
    effective_cfg = {**cfg, "DO_EVAL": do_eval}
    train_dataset = build_cross_encoder_pretrain_dataset(train_pairs_df)
    eval_dataset = build_cross_encoder_pretrain_dataset(eval_pairs_df) if do_eval else train_dataset.select(range(min(len(train_dataset), 1)))

    model = CrossEncoder(
        cross_encoder_model_name,
        device=cfg["DEVICE"],
        max_length=int(cfg["MAX_SEQ_LENGTH"]),
        num_labels=1,
    )
    positive_count = max(int((train_pairs_df["label"] > 0).sum()), 1)
    negative_count = max(int((train_pairs_df["label"] <= 0).sum()), 1)
    pos_weight = torch.tensor(float(negative_count / positive_count), device=model.model.device)
    loss = BinaryCrossEntropyLoss(model=model, pos_weight=pos_weight)
    training_args = build_dictionary_pretrain_training_arguments(checkpoints_dir, effective_cfg)
    eval_callback = DictionaryPretrainEvalCallback(
        dev_pairwise_df=eval_pairs_df,
        cfg=effective_cfg,
        best_model_dir=best_model_dir,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        callbacks=[eval_callback],
    )

    with log_timed(logger, f"Dictionary pretraining for {cfg['EPOCHS']} epochs"):
        trainer.train()

    best_checkpoint_dir = trainer.state.best_model_checkpoint
    if not best_model_dir.exists():
        model.save(str(best_model_dir), safe_serialization=True)
        best_checkpoint_dir = str(best_model_dir)

    history_df, best_epoch, best_metrics = _finalize_dictionary_pretrain_result(
        trainer=trainer,
        eval_callback=eval_callback,
    )
    return {
        "train_pair_examples_df": train_pairs_df,
        "eval_pair_examples_df": eval_pairs_df,
        "history_df": history_df,
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_metrics": best_metrics,
        "best_checkpoint_dir": str(best_checkpoint_dir),
        "best_model_dir": str(best_model_dir),
        "used_eval_split": bool(do_eval),
    }
