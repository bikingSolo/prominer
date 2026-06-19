"""Cross-encoder loading and JSON persistence helpers."""

import json
from pathlib import Path
from typing import Dict

import torch
from sentence_transformers import CrossEncoder


def _resolve_model_torch_dtype(torch_dtype):
    if torch_dtype is None:
        return None
    if isinstance(torch_dtype, str):
        normalized = torch_dtype.strip().lower()
        if normalized in {"", "none"}:
            return None
        if normalized == "auto":
            return "auto"
        if not hasattr(torch, normalized):
            raise ValueError(f"Unsupported TORCH_DTYPE value: {torch_dtype!r}")
        return getattr(torch, normalized)
    return torch_dtype


def build_cross_encoder_load_kwargs(cfg: Dict) -> Dict:
    """Build keyword arguments for loading a cross-encoder."""
    load_kwargs = {
        "trust_remote_code": bool(cfg.get("TRUST_REMOTE_CODE", False)),
        "local_files_only": bool(cfg.get("LOCAL_FILES_ONLY", False)),
    }
    resolved_torch_dtype = _resolve_model_torch_dtype(cfg.get("TORCH_DTYPE"))
    if resolved_torch_dtype is not None:
        load_kwargs["model_kwargs"] = {"torch_dtype": resolved_torch_dtype}
    return load_kwargs


def load_cross_encoder_model_with_config(model_path: str, *, device: str, cfg: Dict) -> CrossEncoder:
    """Load a cross-encoder model with project defaults."""
    return CrossEncoder(
        model_path,
        device=device,
        num_labels=1,
        **build_cross_encoder_load_kwargs(cfg),
    )


def save_json(data: Dict, output_path) -> Path:
    """Save JSON data with UTF-8 encoding."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
