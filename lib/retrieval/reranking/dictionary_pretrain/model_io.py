"""Load cross-encoder models used by dictionary pretraining experiments."""

from pathlib import Path

from sentence_transformers import CrossEncoder


def load_cross_encoder_from_pretrained(model_name_or_path: str, *, device: str) -> CrossEncoder:
    """Load a cross-encoder from a local path or model identifier."""
    model_path = Path(model_name_or_path)
    resolved_model = str(model_path) if model_path.exists() else str(model_name_or_path)
    return CrossEncoder(resolved_model, device=device, num_labels=1)
