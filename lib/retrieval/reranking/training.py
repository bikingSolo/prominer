"""Cross-encoder reranker training loop and callbacks."""

import inspect
import logging
import shutil
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss, LambdaLoss, LambdaRankScheme, ListNetLoss
from transformers import TrainerCallback

from ..tuning import evaluate_dev_predictions
from .candidate_cache import _ensure_candidate_text_map
from .inference import rerank_from_candidate_cache
from .io import build_cross_encoder_load_kwargs
from .training_data import (
    build_cross_encoder_dataset,
    build_cross_encoder_listwise_dataset,
    build_cross_encoder_pairwise_training_data,
    build_cross_encoder_training_data,
)
from lib.utils.logging_utils import log_timed


logger = logging.getLogger(__name__)


def configure_cross_encoder_training_precision(cfg: Dict) -> None:
    """Apply precision settings for cross-encoder training."""
    use_fp16 = bool(cfg.get("USE_FP16", False))
    use_bf16 = bool(cfg.get("USE_BF16", False))
    allow_tf32 = bool(cfg.get("ALLOW_TF32", False))

    if use_fp16 and use_bf16:
        raise ValueError("USE_FP16 and USE_BF16 cannot both be enabled.")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    logger.info(
        "Configured cross-encoder training precision: use_fp16=%s, use_bf16=%s, allow_tf32=%s",
        use_fp16,
        use_bf16,
        allow_tf32,
    )


def _resolve_dev_rerank_topk(cfg: Dict) -> int:
    topk = cfg.get("DEV_RETURN_TOPK", cfg.get("DEV_TOPK", cfg.get("DEV_CANDIDATE_POOL_SIZE")))
    if topk is None:
        raise KeyError("DEV_RETURN_TOPK")
    return int(topk)


class CrossEncoderRerankingEvalCallback(TrainerCallback):
    """Evaluate reranking quality during cross-encoder training."""
    def __init__(
        self,
        *,
        dev_df: pd.DataFrame,
        dev_candidate_cache: Dict[str, Dict],
        mention_column: str,
        candidate_text_map: Dict[tuple[str, str], str] | None,
        cfg: Dict,
        best_model_dir,
    ):
        self.dev_df = dev_df
        self.dev_candidate_cache = dev_candidate_cache
        self.mention_column = str(mention_column)
        self.candidate_text_map = _ensure_candidate_text_map(candidate_text_map)
        self.cfg = cfg
        self.best_model_dir = Path(best_model_dir)
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

        predictions_df = rerank_from_candidate_cache(
            data_df=self.dev_df,
            retriever_cache=self.dev_candidate_cache,
            cross_encoder_model=model,
            mention_column=self.mention_column,
            candidate_text_map=self.candidate_text_map,
            batch_size=int(self.cfg["RERANK_BATCH_SIZE"]),
            topk=_resolve_dev_rerank_topk(self.cfg),
        )
        dev_metrics = evaluate_dev_predictions(predictions_df=predictions_df, data_df=self.dev_df)
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
            model.save(str(self.best_model_dir), safe_serialization=True)
            logger.info(
                "Saved new best cross-encoder model at epoch=%d to %s with %s=%.6f",
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


def build_cross_encoder_training_arguments(output_dir, cfg: Dict):
    """Build training arguments for reranker fine-tuning."""
    training_kwargs = dict(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="steps" if int(cfg.get("DEV_LOSS_EVAL_STEPS", 0)) > 0 else "epoch",
        eval_steps=int(cfg["DEV_LOSS_EVAL_STEPS"]) if int(cfg.get("DEV_LOSS_EVAL_STEPS", 0)) > 0 else None,
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
        lr_scheduler_type=str(cfg.get("LR_SCHEDULER_TYPE", "linear")),
        fp16=bool(cfg.get("USE_FP16", False)),
        bf16=bool(cfg.get("USE_BF16", False)),
        seed=int(cfg["SEED"]),
        remove_unused_columns=False,
        report_to=[],
        dataloader_drop_last=False,
    )
    signature = inspect.signature(CrossEncoderTrainingArguments.__init__)
    if "tf32" in signature.parameters:
        training_kwargs["tf32"] = bool(cfg.get("ALLOW_TF32", False))
    if "save_safetensors" in signature.parameters:
        training_kwargs["save_safetensors"] = True
    return CrossEncoderTrainingArguments(**training_kwargs)


def _resolve_lambdaloss_weighting_scheme(scheme_name: str | None):
    normalized = "ndcg2pp" if scheme_name is None else str(scheme_name).strip().lower()
    if normalized in {"ndcg2pp", "ndcg_loss2pp", "default"}:
        return None
    if normalized in {"ndcg2", "ndcg_loss2"}:
        from sentence_transformers.cross_encoder.losses import NDCGLoss2Scheme

        return NDCGLoss2Scheme()
    if normalized in {"ndcg1", "ndcg_loss1"}:
        from sentence_transformers.cross_encoder.losses import NDCGLoss1Scheme

        return NDCGLoss1Scheme()
    if normalized in {"lambdarank", "lambda_rank"}:
        return LambdaRankScheme()
    if normalized in {"none", "no_weighting"}:
        from sentence_transformers.cross_encoder.losses import NoWeightingScheme

        return NoWeightingScheme()
    raise ValueError(
        f"Unsupported LAMBDALOSS_WEIGHTING_SCHEME: {scheme_name!r}. "
        "Expected one of ['ndcg2pp', 'ndcg2', 'ndcg1', 'lambdarank', 'none']."
    )


def _resolve_cross_encoder_loss(
    *,
    model: CrossEncoder,
    train_pairs_df: pd.DataFrame,
    cfg: Dict,
):
    loss_name = str(cfg.get("LOSS_NAME", "bce")).strip().lower()
    if loss_name == "bce":
        positive_count = max(int((train_pairs_df["label"] > 0).sum()), 1)
        negative_count = max(int((train_pairs_df["label"] <= 0).sum()), 1)
        pos_weight = torch.tensor(float(negative_count / positive_count), device=model.model.device)
        return BinaryCrossEntropyLoss(
            model=model,
            pos_weight=pos_weight,
        )
    if loss_name == "listnet":
        return ListNetLoss(model=model)
    if loss_name == "lambdaloss":
        weighting_scheme = _resolve_lambdaloss_weighting_scheme(cfg.get("LAMBDALOSS_WEIGHTING_SCHEME", "ndcg2pp"))
        k = cfg.get("LAMBDALOSS_K", 1)
        mini_batch_size = cfg.get("LAMBDALOSS_MINI_BATCH_SIZE")
        return LambdaLoss(
            model=model,
            weighting_scheme=weighting_scheme,
            k=None if k is None else int(k),
            sigma=float(cfg.get("LAMBDALOSS_SIGMA", 1.0)),
            reduction_log=str(cfg.get("LAMBDALOSS_REDUCTION_LOG", "binary")),
            mini_batch_size=None if mini_batch_size is None else int(mini_batch_size),
        )
    raise ValueError(
        f"Unsupported cross-encoder loss: {loss_name!r}. "
        "Expected one of ['bce', 'listnet', 'lambdaloss']."
    )


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


def train_cross_encoder_reranker(
    *,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    cross_encoder_model_name: str,
    output_dir,
    cfg: Dict,
    train_candidate_cache: Dict[str, Dict],
    dev_candidate_cache: Dict[str, Dict],
    mention_column: str = "text",
    candidate_text_map: Dict[tuple[str, str], str] | None = None,
) -> Dict[str, object]:
    """Train the final cross-encoder reranker."""
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"

    configure_cross_encoder_training_precision(cfg)

    model = CrossEncoder(
        cross_encoder_model_name,
        device=cfg["DEVICE"],
        max_length=int(cfg["MAX_SEQ_LENGTH"]),
        num_labels=1,
        **build_cross_encoder_load_kwargs(cfg),
    )
    train_examples_df = build_cross_encoder_training_data(
        train_df=train_df,
        retriever_cache=train_candidate_cache,
        vocab_df=vocab_df,
        mention_column=mention_column,
        candidate_text_map=candidate_text_map,
        seed=int(cfg["SEED"]),
    )
    dev_examples_df = build_cross_encoder_training_data(
        train_df=dev_df,
        retriever_cache=dev_candidate_cache,
        vocab_df=vocab_df,
        mention_column=mention_column,
        candidate_text_map=candidate_text_map,
        seed=int(cfg["SEED"]),
    )
    train_pairs_df = build_cross_encoder_pairwise_training_data(train_examples_df)
    dev_pairs_df = build_cross_encoder_pairwise_training_data(dev_examples_df) if len(dev_examples_df) else train_pairs_df
    loss_name = str(cfg.get("LOSS_NAME", "bce")).strip().lower()

    if loss_name in {"listnet", "lambdaloss"}:
        train_dataset = build_cross_encoder_listwise_dataset(train_examples_df)
        eval_dataset = build_cross_encoder_listwise_dataset(dev_examples_df) if len(dev_examples_df) else train_dataset
    else:
        train_dataset = build_cross_encoder_dataset(train_pairs_df)
        eval_dataset = build_cross_encoder_dataset(dev_pairs_df) if len(dev_pairs_df) else train_dataset
    loss = _resolve_cross_encoder_loss(
        model=model,
        train_pairs_df=train_pairs_df,
        cfg=cfg,
    )
    training_args = build_cross_encoder_training_arguments(checkpoints_dir, cfg)
    eval_callback = CrossEncoderRerankingEvalCallback(
        dev_df=dev_df,
        dev_candidate_cache=dev_candidate_cache,
        mention_column=mention_column,
        candidate_text_map=candidate_text_map,
        cfg=cfg,
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

    with log_timed(logger, f"Cross-encoder training for {cfg['EPOCHS']} epochs"):
        trainer.train()

    if not best_model_dir.exists():
        logger.info("No best model was saved during callbacks; saving final model to %s", best_model_dir)
        model.save(str(best_model_dir), safe_serialization=True)

    history_df, best_epoch, best_metrics = _finalize_training_result(
        trainer=trainer,
        eval_callback=eval_callback,
        cfg=cfg,
    )

    return {
        "train_examples_df": train_examples_df,
        "train_pair_examples_df": train_pairs_df,
        "history_df": history_df,
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "stopped_early": bool(eval_callback.stopped_early),
        "stop_epoch": None if eval_callback.stop_epoch is None else int(eval_callback.stop_epoch),
        "best_checkpoint_dir": str(best_model_dir),
        "best_model_dir": str(best_model_dir),
    }
