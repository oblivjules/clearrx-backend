"""Explicit configuration for genuine local TrOCR and truthful registry metadata."""
import os
from pathlib import Path


def local_model_configuration():
    base = os.getenv("TROCR_BASE_MODEL", "microsoft/trocr-base-handwritten").strip()
    if not base:
        raise ValueError("TROCR_BASE_MODEL must identify a local-compatible TrOCR model")
    configured = os.getenv("TROCR_CHECKPOINT_DIR", "").strip()
    if configured:
        checkpoint = Path(configured).expanduser().resolve()
        if not checkpoint.is_dir():
            raise ValueError("TROCR_CHECKPOINT_DIR does not exist")
    else:
        runs = Path(__file__).resolve().parent / "runs" / "trocr-rx"
        checkpoint = next((p for p in [runs / "best", runs / "checkpoint-epoch-20"] if p.is_dir()), None)
    return dict(base_model=base, weights_path=str(checkpoint) if checkpoint else "n/a",
                fine_tuned_on="unverified local checkpoint" if checkpoint else "n/a")
