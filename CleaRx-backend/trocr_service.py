"""Local TrOCR recognition and prescription parsing entry points."""
from __future__ import annotations

import numpy as np


def run_trocr_ocr(image: np.ndarray) -> tuple[str, float]:
    """Recognize handwriting using the local TrOCR service."""
    from ocr_service import run_trocr

    return run_trocr(image)


def run_trocr_parse(raw_text: str) -> list[dict]:
    """Structure recognized text using the prescription parser."""
    from parser import parse_prescription

    return parse_prescription(raw_text)
