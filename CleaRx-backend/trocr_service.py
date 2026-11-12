"""Recognition provider adapter. Product-facing names are resolved separately.

The compatibility entry point run_trocr_ocr invokes the configured provider;
its name is not evidence that a local TrOCR model executed.
"""
from __future__ import annotations

import base64
import os
import time

import cv2
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────

_ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

LOCAL_MODEL = "local-trocr"
PRIMARY_OPTION = "clearx-trocr"
PRIMARY_LABEL = "CleaRx TrOCR"
EXTERNAL_MODEL = "claude-sonnet-5-5"
EXTERNAL_LABEL = "Claude Sonnet 5.5"


def resolve_model(option: str) -> tuple[str, str]:
    if option == PRIMARY_OPTION:
        return LOCAL_MODEL, PRIMARY_LABEL
    if option == EXTERNAL_MODEL:
        return EXTERNAL_MODEL, EXTERNAL_LABEL
    raise ValueError("Unsupported recognition option")

SUPPORTED_MODELS = {
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
}


def validate_model(model: str) -> str:
    if model not in SUPPORTED_MODELS:
        raise ValueError("Unsupported recognition option")
    return model

# Retry settings for HTTP 429 (rate limit) responses
_MAX_RETRIES    = 3
_BACKOFF_BASE_S = 2.0   # Three attempts permit waits of 2s and 4s.

_client = None   # anthropic.Anthropic handle


def _init() -> bool:
    """Lazily initialise the Anthropic client. Returns False when unavailable."""
    global _client
    if _client is not None:
        return True
    if not _ANTHROPIC_API_KEY:
        return False
    try:
        import anthropic  # type: ignore
        # One explicit retry policy; avoid multiplying SDK retries by our loop.
        _client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY, max_retries=0, timeout=45.0)
        return True
    except ImportError:
        return False
    except Exception:
        return False


def _message_with_retry(system: str, messages: list[dict], model: str = EXTERNAL_MODEL) -> str | None:
    """
    Call _client.messages.create with exponential backoff on 429.
    Returns the response text, or None on persistent failure.
    """
    validate_model(model)
    for attempt in range(_MAX_RETRIES):
        try:
            response = _client.messages.create(
                model=model,
                max_tokens=2048,
                system=system,
                messages=messages,
            )
            return "\n".join(block.text for block in response.content
                             if block.type == "text").strip()
        except Exception as exc:
            is_rate_limit = getattr(exc, 'status_code', None) in {429, 529}
            if is_rate_limit and attempt < _MAX_RETRIES - 1:
                wait = _BACKOFF_BASE_S ** (attempt + 1)
                time.sleep(wait)
                continue
            return None
    return None


# ── Vision OCR ─────────────────────────────────────────────────────────────────

_OCR_SYSTEM = (
    "You are an expert medical transcriptionist specializing in Philippine "
    "doctor handwriting. You excel at reading cursive, connected, and informal "
    "handwriting on real-world prescription paper, including crumpled, folded, "
    "or low-contrast images."
)

_OCR_PROMPT = (
    "This is a Philippine doctor's handwritten prescription photo. The image may "
    "include a printed clinic header, patient information fields, and a doctor "
    "signature block — IGNORE all of that printed/stamped content. Focus only on "
    "the handwritten drug name(s) and instruction lines written in ink by the doctor.\n\n"
    "The handwritten drug/instruction section typically follows this structure:\n"
    "  Line 1: Drug name (brand or generic) with optional strength, "
    "e.g. 'Amoxicillin 500mg' or 'Orahex' or 'Cefuroxime 250mg/5ml'\n"
    "  Line 2: Sig/instructions — how to take it, "
    "e.g. 'S: 1 tab TID' or 'Rinse 5ml for 1 minute' or '1x3 x7d'\n"
    "  Optional further lines: duration, refills, e.g. 'x 7 days', 'for 5 days', '#14'\n\n"
    "Common Philippine prescription abbreviations:\n"
    "  FREQUENCY — OD=once daily, BD/BID=twice daily, TID/TDS=three times daily,\n"
    "    QID/QDS=four times daily, PRN/SOS=as needed, HS/QN=at bedtime,\n"
    "    Q4H/Q6H/Q8H/Q12H=every 4/6/8/12 hours, STAT=immediately,\n"
    "    Mane/OM/QAM=every morning, Nocte/QN/QPM=every night,\n"
    "    EOD/QOD=every other day, QWK=once weekly, BIW=twice weekly, TIW=three times weekly,\n"
    "    Ad lib=as desired (= PRN)\n"
    "  MEAL — PC/p.c.=after meals, AC/a.c.=before meals, CC/c.c.=with meals\n"
    "  ROUTE — PO=by mouth, IV=intravenous, IVTT=IV through tubing, IM=intramuscular,\n"
    "    SC/SQ=subcutaneous, SL=sublingual, PR=rectal, PV=vaginal, NGT=nasogastric tube,\n"
    "    Top/Topical, Inh=inhalation, Ophth=ophthalmic\n"
    "  DOSE FORM — Tab=tablet, Cap=capsule, Gtt/Gtts=drop/drops, Ung=ointment,\n"
    "    Susp=suspension, Sol=solution, Elix=elixir, Supp=suppository, Lot=lotion\n"
    "  QUANTITY — SS=½ (semisse=half), #N or Mitte N=dispense N units\n"
    "  N×M shorthand: '1×1'=1 tab once daily, '1×3'=1 tab three times daily\n"
    "  N-N-N shorthand: '1-0-1'=twice daily, '1-1-1'=three times daily\n\n"
    "CURSIVE HANDWRITING — common confusions to watch for:\n"
    "  - 'rn' strokes next to each other often look like a single 'm'\n"
    "  - 'u' and 'n' look nearly identical; 'e' and 'i' are often ambiguous\n"
    "  - 'cl' can look like 'd'; 'vv' can look like 'w'\n"
    "  - Drug names may end in: -cillin, -mycin, -statin, -pril, -sartan, -zole, "
    "-olol, -dipine, -dine, -hex, -floxacin, -oxacin, -vir, -mab, -nib, -tidine\n"
    "  - Antiseptics and mouthwashes: Orahex, Chlorhexidine, Betadine, Povidone\n"
    "  - Liquid/topical instructions use verbs: Rinse, Gargle, Apply, Instill\n"
    "  - Volume dosing: '5ml', '10ml', 'cc' — not tablet counts\n\n"
    "IMAGE QUALITY — how to handle a crumpled or folded prescription:\n"
    "  - Crease lines from folded paper are physical folds, NOT handwritten text — ignore them\n"
    "  - Low-contrast areas: look for ink colour/shade slightly darker than the paper\n"
    "  - Text cut off at the image edge: transcribe what is visible, append [CUTOFF]\n\n"
    "Rules:\n"
    "- Output one line of text per visible handwritten line in the drug/instruction section.\n"
    "- Do NOT transcribe printed clinic header, patient name, age, date, or doctor signature.\n"
    "- If a word is illegible and you are less than 85% confident, write [UNCERTAIN] — do NOT guess.\n"
    "- CRITICAL: Do NOT use medical knowledge to fill in, infer, or complete any dosage, "
    "frequency, or drug information. Transcribe ONLY what is literally visible in the image. "
    "If you cannot read a value, write [UNCERTAIN].\n"
    "- Output ONLY the raw transcribed text. No commentary, no headers, no markdown."
)


def run_trocr_ocr(image: np.ndarray, model: str = EXTERNAL_MODEL) -> tuple[str, float | None] | None:
    """
    Send a BGR/grayscale prescription image to Claude Vision.

    Returns (text, None): the provider does not supply calibrated OCR confidence,
    or None when the API is unavailable or the request fails.
    """
    validate_model(model)
    if not _init():
        return None

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image

    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    img_bytes = buf.tobytes()
    img_b64   = base64.standard_b64encode(img_bytes).decode("ascii")

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": _OCR_PROMPT,
                    },
                ],
            }
        ]

        text = _message_with_retry(_OCR_SYSTEM, messages, model)

        if not text:
            return None

        return text, None

    except Exception:
        return None


# ── Structured parser ──────────────────────────────────────────────────────────


def run_trocr_parse(raw_text: str) -> list[dict]:
    """Use the same deterministic parser for every recognition source."""
    from parser import parse_prescription

    return parse_prescription(raw_text)
