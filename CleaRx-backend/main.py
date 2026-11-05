from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
import uuid
import cv2
import os
import numpy as np
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logger = logging.getLogger(__name__)

# Load configuration before importing services that read environment variables at import time.
load_dotenv(Path(__file__).resolve().parent / ".env")

from image_processor import ImageProcessor
from trocr_service import run_trocr_ocr
from drug_validator import validate_drug_name, DrugValidator
from parser import parse_prescription, refresh_drug_catalog, _drug_name_passes_sanity

try:
    from db import check_database_connection as _check_database_connection
    from db import fetch_drug_catalog as _fetch_drug_catalog
    from db import log_validate_capture as _db_log_validate_capture
    from db import create_capture_session as _db_create_capture_session
    from db import create_prescription_image as _db_create_prescription_image
    from db import update_prescription_image_status as _db_update_image_status
    from db import prescription_image_exists as _db_prescription_image_exists
    from db import has_completed_transcription_result as _db_has_completed_transcription_result
    from db import close_capture_session as _db_close_capture_session
    from db import log_ocr_evaluation as _db_log_ocr_evaluation
    from db import log_drug_validation_evaluation as _db_log_drug_validation
    from db import create_preprocessing_log as _db_create_preprocessing_log
    from db import get_or_create_trocr_model as _db_get_or_create_trocr_model
    from db import create_inference_job as _db_create_inference_job
    from db import finish_inference_job as _db_finish_inference_job
    from db import create_transcription_output_with_fields as _db_create_transcription_output_with_fields
    from db import persist_drug_matches as _db_persist_drug_matches
    from db import persist_structured_prescription as _db_persist_structured_prescription
    from confidence_config import (
        LOW_MATCH_CONFIDENCE, LOW_OCR_CONFIDENCE, LINE_LOW_CONFIDENCE,
        LINE_MATCHED, LINE_UNMATCHED, STATUS_EXTRACTED,
        STATUS_EXTRACTED_WITH_FLAGS, STATUS_NO_ITEMS, UNCERTAIN_FIELD_SCORE_CAP,
        confidence_tier,
    )
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

try:
    from rapidfuzz.distance import Levenshtein as _CER_BACKEND
    _CER_BACKEND_NAME = "rapidfuzz"
except ImportError:
    try:
        import Levenshtein as _LevenshteinModule

        class _CER_BACKEND:  # thin shim so both backends expose .editops(ref, hyp)
            @staticmethod
            def editops(ref, hyp):
                return _LevenshteinModule.editops(ref, hyp)

        _CER_BACKEND_NAME = "python-Levenshtein"
    except ImportError:
        _CER_BACKEND = None
        _CER_BACKEND_NAME = "difflib"
        print(
            "[CER] WARNING: neither rapidfuzz nor python-Levenshtein is installed — "
            "falling back to difflib, which is slower and not a true edit distance. "
            "Add 'rapidfuzz' to requirements.txt for production."
        )


def _levenshtein_ops(ref: str, hyp: str) -> dict:
    """Edit-op counts (substitutions/deletions/insertions) between two strings."""
    if not ref and not hyp:
        return {"substitutions": 0, "deletions": 0, "insertions": 0, "edit_distance": 0}

    if ref == hyp:
        # Common case (e.g. raw vs. normalized turn out identical) — skip the
        # edit-distance computation entirely.
        return {"substitutions": 0, "deletions": 0, "insertions": 0, "edit_distance": 0}

    if _CER_BACKEND_NAME == "difflib":
        import difflib
        sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
        subs = dels = ins = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "replace":
                subs += max(i2 - i1, j2 - j1)
            elif tag == "delete":
                dels += i2 - i1
            elif tag == "insert":
                ins += j2 - j1
        dist = subs + dels + ins
    else:
        editops = _CER_BACKEND.editops(ref, hyp)
        subs = dels = ins = 0
        for op in editops:
            tag = op.tag if hasattr(op, "tag") else op[0]
            if tag == "replace":
                subs += 1
            elif tag == "delete":
                dels += 1
            elif tag == "insert":
                ins += 1
        dist = len(editops)

    return {"substitutions": subs, "deletions": dels, "insertions": ins, "edit_distance": dist}


def _compute_cer(reference: str, hypothesis: str) -> dict:
    """
    Compute Character Error Rate (CER) using Levenshtein edit distance.

    Two variants are ALWAYS computed, so normalization can never silently
    hide a real transcription error:

      - raw_cer / raw_cer_percent:
            strings compared as-typed (only .strip() applied). This is the
            literal, unforgiving accuracy — every case, spacing, and
            punctuation difference counts as an error.

      - cer / cer_percent (normalized — used for pass/fail):
            lowercased, punctuation collapsed to whitespace, whitespace
            collapsed. Formatting artifacts ("#5" vs "5", "500mg/tab" vs
            "500mg tab", "TAB" vs "tab") are not transcription errors and
            shouldn't count as ones for Objective 2.2 — but a genuinely
            missing/garbled/extra WORD still shows up as substitutions,
            deletions or insertions here, same as before. Normalization
            only removes formatting noise; it never removes content.

    `normalization_gap_percent` = raw_cer_percent - cer_percent. A large gap
    means normalization is doing a lot of work on a given sample — worth a
    manual look even if the normalized result passes, since it means the
    ground truth and hypothesis differ heavily in formatting even where
    content agrees.

    CER = (S + D + I) / N, where N = length of that variant's reference.
    Objective 2.2 target: normalized CER <= 0.20 (20%).
    """
    def normalize(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)  # punctuation/symbols -> space, not a literal char
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    raw_ref = (reference or "").strip()
    raw_hyp = (hypothesis or "").strip()
    norm_ref = normalize(reference or "")
    norm_hyp = normalize(hypothesis or "")

    if not norm_ref:
        return {
            "cer": 0.0, "cer_percent": 0.0,
            "substitutions": 0, "deletions": 0, "insertions": 0,
            "edit_distance": 0, "ref_length": 0, "hyp_length": len(norm_hyp),
            "raw_cer": 0.0, "raw_cer_percent": 0.0,
            "normalization_gap_percent": 0.0,
            "passes_objective": False,
        }

    norm_ops = _levenshtein_ops(norm_ref, norm_hyp)
    norm_cer = norm_ops["edit_distance"] / len(norm_ref)

    # Skip the second edit-distance pass when normalization changed nothing
    # (no case/punctuation/whitespace formatting differences to strip) —
    # raw and normalized results are identical in that case.
    if raw_ref == norm_ref and raw_hyp == norm_hyp:
        raw_ops = norm_ops
    else:
        raw_ops = _levenshtein_ops(raw_ref, raw_hyp)
    raw_cer = (raw_ops["edit_distance"] / len(raw_ref)) if raw_ref else 0.0

    return {
        "cer": round(norm_cer, 4),
        "cer_percent": round(norm_cer * 100, 2),
        "substitutions": norm_ops["substitutions"],
        "deletions": norm_ops["deletions"],
        "insertions": norm_ops["insertions"],
        "edit_distance": norm_ops["edit_distance"],
        "ref_length": len(norm_ref),
        "hyp_length": len(norm_hyp),
        "raw_cer": round(raw_cer, 4),
        "raw_cer_percent": round(raw_cer * 100, 2),
        "normalization_gap_percent": round((raw_cer - norm_cer) * 100, 2),
        "passes_objective": norm_cer <= 0.20,
    }


def _check_ground_truth_drug_name(ground_truth: str) -> dict | None:
    """
    Best-effort sanity check on the QA-typed ground truth (not the OCR output):
    does its apparent drug name exist in the validated drug catalog? Catches
    ground-truth typos (e.g. 'Azthrimycin' instead of 'Azithromycin') that
    would otherwise silently skew CER against a correct OCR result.

    Diagnostic only — logged as a warning and returned for visibility. Never
    alters cer/passes_objective and never blocks persistence.
    """
    if not ground_truth or not ground_truth.strip():
        return None

    first_line = ground_truth.strip().splitlines()[0].strip()
    candidate = re.split(r"\d", first_line)[0].strip(" -:/")
    if not candidate:
        return None

    try:
        result = validate_drug_name(candidate)
    except Exception as exc:
        print(f"[CER][gt-check] drug name validation failed: {exc}")
        return None

    matched = bool(result.get("matched"))
    if not matched:
        print(
            f"[CER][gt-check] WARNING: ground-truth drug name {candidate!r} "
            f"was not found in the drug catalog. If the OCR hypothesis for "
            f"this sample disagrees with the ground truth on the drug name, "
            f"check whether the GROUND TRUTH has a typo before treating it "
            f"as an OCR error."
        )

    return {
        "candidate": candidate,
        "matched": matched,
        "corrected": result.get("corrected"),
        "confidence": result.get("confidence"),
    }


def _persist_cer(
    *,
    image_id: str | None,
    ground_truth: str,
    hypothesis: str,
    cer_result: dict,
    request_id: str | None = None,
) -> None:
    """Best-effort persistence of a CER record. Never raises."""
    if not _DB_AVAILABLE or cer_result is None:
        return
    logger.info(
        "persistence entry module=ocr_evaluation image_id=%s request_id=%s",
        image_id, request_id,
    )
    if not image_id:
        logger.error("OCR_EVALUATION skipped: missing image_id")
        return
    try:
        evaluation_id = _db_log_ocr_evaluation(
            image_id=image_id,
            ground_truth=ground_truth,
            hypothesis=hypothesis,
            cer=cer_result["cer"],
            cer_percent=cer_result["cer_percent"],
            substitutions=cer_result["substitutions"],
            deletions=cer_result["deletions"],
            insertions=cer_result["insertions"],
            edit_distance=cer_result["edit_distance"],
            ref_length=cer_result["ref_length"],
            hyp_length=cer_result["hyp_length"],
            passes_objective=cer_result["passes_objective"],
            target_cer=0.20,
        )
        logger.info("OCR_EVALUATION row inserted: %s", evaluation_id)
    except Exception as exc:
        logger.error("OCR_EVALUATION insert failed (image_id=%s): %s", image_id, exc)


def _persist_module_2(
    *,
    image_id: str | None,
    preprocessing_id: str | None,
    provider: str,
    raw_text: str,
    confidence: float,
    image_shape: tuple[int, ...],
    medicines: list[dict],
    raw_medicines: list[dict] | None = None,
    inference_time_ms: float,
    failure_reason: str | None,
    request_id: str | None = None,
) -> tuple[str | None, dict[int, str]]:
    if not _DB_AVAILABLE or not preprocessing_id:
        return None, {}
    logger.info(
        "persistence entry module=transcription_output image_id=%s request_id=%s",
        image_id, request_id,
    )

    model_name = provider
    prefix = "TROCR"
    model_id = None
    job_id = None
    try:
        model_id = _db_get_or_create_trocr_model(
            model_name=model_name,
            model_version=os.getenv(f"{prefix}_MODEL_VERSION", "1"),
            base_model=os.getenv(
                f"{prefix}_BASE_MODEL",
                "microsoft/trocr-base-handwritten",
            ),
            fine_tuned_on=os.getenv(
                f"{prefix}_FINE_TUNED_ON",
                "CleaRx prescription dataset",
            ),
            weights_path=os.getenv(
                f"{prefix}_WEIGHTS_PATH",
                "runs/trocr-rx/best",
            ),
            status=os.getenv(f"{prefix}_MODEL_STATUS", "active"),
        )
    except Exception as exc:
        logger.error("trocr_model_registry failed (image_id=%s, step=model): %s", image_id, exc)
        return None, {}

    try:
        job_id = _db_create_inference_job(
            preprocessing_id=preprocessing_id,
            model_id=model_id,
            input_width_px=int(image_shape[1]),
            input_height_px=int(image_shape[0]),
        )
    except Exception as exc:
        logger.error("trocr_inference_jobs submission failed (image_id=%s, step=submit): %s", image_id, exc)
        return None, {}

    try:
        _db_finish_inference_job(
            job_id=job_id,
            inference_time_ms=inference_time_ms,
            failure_reason=failure_reason,
        )
    except Exception as exc:
        logger.error("trocr_inference_jobs completion failed (image_id=%s, step=complete): %s", image_id, exc)
        return None, {}

    output_id = None
    try:
        field_rows = []
        cursor = 0
        field_sequence = 0
        for medicine in raw_medicines or medicines:
            for field_name, value in medicine.items():
                if field_name.startswith("_"):
                    continue
                if value is None:
                    continue
                field_value = str(value)
                start = raw_text.find(field_value, cursor)
                if start < 0:
                    start = raw_text.find(field_value)
                end = start + len(field_value) if start >= 0 else start
                if start >= 0:
                    cursor = end
                field_rows.append((field_name, field_value, field_sequence, float(confidence), start, end))
                field_sequence += 1
        output_id, field_ids = _db_create_transcription_output_with_fields(
            job_id=job_id,
            raw_text=raw_text,
            overall_confidence=float(confidence),
            token_count=len(raw_text.split()),
            output_status="failed" if failure_reason else "completed",
            fields=field_rows,
        )
        return output_id, field_ids
    except Exception as exc:
        logger.error("transcription persistence failed (image_id=%s, step=output): %s", image_id, exc)
        return output_id, {}


def _persist_modules_3_4(
    *,
    image_id: str | None,
    output_id: str | None,
    field_ids: dict[int, str],
    medicines: list[dict],
    raw_medicines: list[dict],
    ocr_confidence: float,
    request_id: str | None = None,
) -> None:
    if not _DB_AVAILABLE or not output_id:
        return
    logger.info(
        "persistence entry module=drug_matches_structured image_id=%s request_id=%s",
        image_id, request_id,
    )
    try:
        match_rows = []
        raw_by_index = {
            index: value for index, value in enumerate(raw_medicines)
        }
        field_sequence = 0
        for index, medicine in enumerate(raw_medicines):
            raw_drug = medicine.get("_drug_name_raw") or medicine.get("drug_name")
            if not raw_drug:
                field_sequence += sum(
                    value is not None and not name.startswith("_")
                    for name, value in medicine.items()
                )
                continue
            result = (
                {}
                if medicine.get("_drug_name_is_class")
                else validate_drug_name(str(raw_drug))
            )
            _persist_drug_validation(
                image_id=image_id,
                result=result,
                request_id=request_id,
            )
            matched_name = result.get("corrected") if result.get("matched") else None
            fuzzy_score = result.get("confidence")
            flagged = bool(
                ocr_confidence < LOW_OCR_CONFIDENCE
                or not result.get("matched")
                or float(fuzzy_score or 0.0) < LOW_MATCH_CONFIDENCE
            )
            reasons = []
            if not result.get("matched"):
                reasons.append("no_lexicon_match")
            if result.get("matched") and float(fuzzy_score or 0.0) < LOW_MATCH_CONFIDENCE:
                reasons.append("low_match_score")
            if ocr_confidence < LOW_OCR_CONFIDENCE:
                reasons.append("low_ocr_confidence")
            field_id = field_ids.get(field_sequence)
            if field_id:
                match_rows.append({
                    "raw_input": str(raw_drug),
                    "matched_name": matched_name,
                    "raw_score": result.get("raw_score"),
                    "field_id": field_id,
                    "lexicon_id": _LEXICON_ID_BY_NAME.get(str(matched_name or "").lower(), (None, False))[0],
                    "match_type": result.get("match_type"),
                    "is_flagged": flagged,
                    "flag_reason": ",".join(reasons) or None,
                    "original_token": str(raw_drug),
                    "corrected_token": str(matched_name or raw_drug),
                    "token_sort_ratio": result.get("token_sort_ratio"),
                    "partial_ratio": result.get("partial_ratio"),
                    "algorithm_used": result.get("algorithm_used"),
                    "score_cutoff_used": result.get("score_cutoff_used"),
                    "correction_applied": bool(matched_name and matched_name != raw_drug),
                })
            field_sequence += sum(
                value is not None and not name.startswith("_")
                for name, value in raw_by_index[index].items()
            )
        _db_persist_drug_matches(matches=match_rows)
    except Exception as exc:
        logger.error("drug matching persistence failed (image_id=%s, output_id=%s, step=matches): %s", image_id, output_id, exc)

    try:
        line_items = []
        any_flagged = False
        for index, medicine in enumerate(medicines):
            raw = raw_medicines[index] if index < len(raw_medicines) else {}
            raw_drug = raw.get("_drug_name_raw") or raw.get("drug_name")
            result = (
                {}
                if raw.get("_drug_name_is_class")
                else validate_drug_name(str(raw_drug or "")) if raw_drug else {}
            )
            matched = bool(result.get("matched"))
            fuzzy = float(result.get("confidence") or 0.0) if matched else None
            flagged = bool(medicine.get("uncertain"))
            any_flagged = any_flagged or flagged
            field_flags = {
                name: name in (medicine.get("uncertain_fields") or [])
                for name in medicine
            }
            marker_fields = set(medicine.get("_marker_fields") or [])
            fields = []
            for name, value in medicine.items():
                if (
                    name.startswith("_")
                    or name in {
                        "uncertain", "uncertain_fields", "drug_name_raw",
                        "dosage", "duration_source",
                    }
                    or value in (None, "")
                ):
                    continue
                fuzzy_value = fuzzy if name in {"drug_name", "brand_name"} else None
                composite = (
                    0.5 * ocr_confidence + 0.5 * (fuzzy_value or 0.0)
                    if name in {"drug_name", "brand_name"} else ocr_confidence
                )
                if name in marker_fields:
                    composite = min(composite, UNCERTAIN_FIELD_SCORE_CAP)
                fields.append({
                    "field_name": name,
                    "raw_ocr_confidence": ocr_confidence,
                    "fuzzy_match_score": fuzzy_value,
                    "composite_score": composite,
                    "confidence_tier": confidence_tier(composite),
                    "is_flagged": bool(field_flags.get(name, False)),
                })
            line_items.append({
                "item_order": index + 1,
                "drug_name_raw": raw_drug,
                "drug_name_matched": medicine.get("drug_name") if matched else None,
                "lexicon_id": _LEXICON_ID_BY_NAME.get(str(medicine.get("drug_name") or "").lower(), (None, False))[0],
                "dosage": medicine.get("dosage"),
                "frequency": medicine.get("frequency"),
                "route": medicine.get("route_of_administration"),
                "duration": medicine.get("duration"),
                "quantity": medicine.get("quantity"),
                "line_item_status": LINE_LOW_CONFIDENCE if matched and flagged else LINE_MATCHED if matched else LINE_UNMATCHED,
                "brand_name": medicine.get("brand_name"),
                "strength": medicine.get("strength"),
                "dosage_form": medicine.get("dosage_form"),
                "dose_amount": medicine.get("dose_amount"),
                "timing": medicine.get("timing"),
                "meal_relation": medicine.get("meal_relation"),
                "administration_instructions": medicine.get("administration_instructions"),
                "duration_source": medicine.get("duration_source"),
                "confidence_fields": fields,
            })
        status = STATUS_NO_ITEMS if not line_items else STATUS_EXTRACTED_WITH_FLAGS if any_flagged else STATUS_EXTRACTED
        prescription_id = _db_persist_structured_prescription(
            output_id=output_id,
            prescription_date=None,
            overall_status=status,
            line_items=line_items,
        )
        if prescription_id is None:
            logger.error(
                "structured prescription already exists; skipped duplicate "
                "(image_id=%s, output_id=%s, step=structured)",
                image_id, output_id,
            )
    except Exception as exc:
        logger.error("structured prescription persistence failed (image_id=%s, output_id=%s, step=structured): %s", image_id, output_id, exc)


def _persist_drug_validation(
    *, image_id: str | None, result: dict, request_id: str | None = None
) -> None:
    """Best-effort persistence of one drug-validation result."""
    if not _DB_AVAILABLE or not image_id:
        return
    logger.info(
        "persistence entry module=drug_match_evaluation image_id=%s request_id=%s",
        image_id, request_id,
    )
    try:
        evaluation_id = _db_log_drug_validation(
            image_id=image_id,
            original_text=str(result.get("original") or ""),
            corrected_text=str(result.get("corrected") or ""),
            match_type=result.get("match_type"),
            confidence=float(result.get("confidence", 0.0) or 0.0),
            matched=bool(result.get("matched")),
        )
        if evaluation_id is None:
            logger.info(
                "DRUG_MATCH_EVALUATION skipped duplicate (image_id=%s, original_text=%s)",
                image_id, result.get("original"),
            )
        else:
            print(f"[db] DRUG_MATCH_EVALUATION row inserted: {evaluation_id}")
    except Exception as exc:
        print(f"[db] drug validation insert failed: {exc}")


# ── Service singletons ─────────────────────────────────────────────────────────
_img_proc = ImageProcessor()
_LEXICON_ID_BY_NAME: dict[str, tuple[str, bool]] = {}


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] CleaRx Backend v12.0.0")
    print(f"[startup] ImageProcessor   : ready")
    print("[startup] OCRService       : (trocr_service)")
    if _DB_AVAILABLE:
        try:
            _check_database_connection()
            drug_catalog = _fetch_drug_catalog()
            global _LEXICON_ID_BY_NAME
            _LEXICON_ID_BY_NAME = {}
            duplicate_names: set[str] = set()
            for entry in drug_catalog:
                drug_id, name, _, discontinued = entry
                key = name.strip().lower()
                if key in _LEXICON_ID_BY_NAME:
                    duplicate_names.add(key)
                if key not in _LEXICON_ID_BY_NAME or (
                    not discontinued and _LEXICON_ID_BY_NAME[key][1]
                ):
                    _LEXICON_ID_BY_NAME[key] = (str(drug_id), bool(discontinued))
            if duplicate_names:
                logger.warning(
                    "duplicate drug lexicon names resolved deterministically: %s",
                    sorted(duplicate_names),
                )
            catalog_size = DrugValidator.use_database_catalog(drug_catalog)
            if catalog_size:
                refresh_drug_catalog()
                print(f"[startup] Drug catalog: {catalog_size} names loaded from Supabase")
            else:
                print("[startup] Drug catalog: empty; using legacy local lexicon")
            print("[startup] Supabase database: connected")
        except Exception as exc:
            print(f"[startup] Supabase database: connection/catalog failed ({exc})")
    else:
        print("[startup] Supabase database: disabled (database module unavailable)")
    yield
    print("[shutdown] CleaRx Backend stopping")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="CleaRx Backend", version="12.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/preview-check")
async def preview_check(file: UploadFile = File(...)):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"readable": False, "score": 0, "instruction": "Camera error — please retry."}

    lighting = _img_proc.check_lighting(img)
    if lighting == "good":
        lighting = _img_proc.check_lighting_cnr(img)

    if lighting == "too_dark":
        return {"readable": False, "score": 0,
                "instruction": "Too dark — move to a brighter spot."}
    if lighting == "too_bright":
        return {"readable": False, "score": 0,
                "instruction": "Too bright — tilt slightly to reduce glare."}

    blur = _img_proc.check_blur(img)
    if blur == "blurry":
        return {"readable": False, "score": 0,
                "instruction": "Hold still — text is blurry."}

    score     = _img_proc._fast_readability_score(img)
    hw        = _img_proc.detect_handwriting(img)
    text_type = "handwritten" if hw["is_handwritten"] else "printed"

    effective_threshold = 55
    if hw["is_handwritten"] and hw["confidence"] >= 0.7:
        score = min(score + 8, 100)
        effective_threshold = 50
    elif text_type == "printed":
        effective_threshold = 65

    if score >= effective_threshold:
        instruction = "Text looks clear — tap capture!" if score >= 75 \
                      else "Mostly readable — tap capture or hold steadier."
        return {"readable": True, "score": score, "instruction": instruction,
                "text_type": text_type, "handwriting_confidence": hw["confidence"]}
    elif score >= 35:
        return {"readable": False, "score": score,
                "instruction": "Move closer or improve lighting for clearer text.",
                "text_type": text_type, "handwriting_confidence": hw["confidence"]}
    else:
        return {"readable": False, "score": score,
                "instruction": "Text not clear enough — align and hold still.",
                "text_type": text_type, "handwriting_confidence": hw["confidence"]}


@router.get("/health")
def health():
    return {"status": "ok", "version": "12.0.0"}


@router.post("/capture-session")
def create_capture_session(
    user_id: str = Form(...),
    device_model: str | None = Form(None),
    os_version: str | None = Form(None),
    app_version: str | None = Form(None),
    image_source: str = Form("camera"),
):
    if not _DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="database unavailable")

    try:
        session_id = _db_create_capture_session(
            user_id=user_id,
            device_model=device_model,
            os_version=os_version,
            app_version=app_version,
            image_source=image_source,
        )
        return {"status": "created", "session_id": session_id}
    except Exception as exc:
        print(f"[db] capture-session creation failed: {exc}")
        raise HTTPException(status_code=503, detail="capture session creation failed")


@router.post("/capture-image")
async def create_capture_image(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    exposure_time: float | None = Form(None),
    iso_sensitivity: int | None = Form(None),
    focal_length: float | None = Form(None),
):
    if not _DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="database unavailable")

    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"status": "error", "reason": "invalid_image"}

    height, width = img.shape[:2]
    try:
        image_id, attempt_number = _db_create_prescription_image(
            session_id=session_id,
            image_bytes=raw,
            width=width,
            height=height,
            exposure_time=exposure_time,
            iso_sensitivity=iso_sensitivity,
            focal_length=focal_length,
        )
        return {"status": "created", "image_id": image_id, "attempt_number": attempt_number}
    except Exception as exc:
        logger.error("capture-image creation failed (session_id=%s): %s", session_id, exc)
        raise HTTPException(status_code=503, detail="capture image creation failed")


@router.post("/capture-session/close")
def close_capture_session(
    session_id: str = Form(...),
    session_status: str = Form("completed"),
):
    if not _DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="database unavailable")
    if session_status not in {"completed", "abandoned"}:
        raise HTTPException(status_code=400, detail="invalid session status")

    try:
        _db_close_capture_session(
            session_id=session_id,
            session_status=session_status,
        )
        return {"status": "closed", "session_id": session_id}
    except Exception as exc:
        print(f"[db] capture-session close failed: {exc}")
        raise HTTPException(status_code=503, detail="capture session close failed")


@router.post("/validate-capture")
async def validate_capture(
    file: UploadFile = File(...),
    image_id: str | None = Form(None),
):
    if _DB_AVAILABLE and image_id:
        try:
            uuid.UUID(image_id)
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(status_code=400, detail="image_id must be a valid UUID")
        try:
            if not _db_prescription_image_exists(image_id=image_id):
                raise HTTPException(status_code=404, detail="image_id was not found")
        except HTTPException:
            raise
        except Exception as exc:
            print(f"[db] image lookup failed: {exc}")
            raise HTTPException(status_code=503, detail="database lookup failed")

    t_start = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {
            "is_usable": False,
            "rejection_reason": "invalid_image",
            "instruction": "Camera error — please retry.",
            "quality_score": 0,
            "feedback": {"lighting": "unknown", "blur": "unknown", "framing": "unknown"},
        }

    lighting = _img_proc.check_lighting(img)
    if lighting == "good":
        lighting = _img_proc.check_lighting_cnr(img)

    blur     = _img_proc.check_blur(img)
    quad     = _img_proc.find_paper_quad(img, min_coverage=0.06)
    img_area = img.shape[0] * img.shape[1]

    framing           = "not_found"
    doc_quad_response = None
    if quad is not None:
        coverage = cv2.contourArea(quad) / img_area
        framing  = "aligned" if coverage >= 0.10 else "too_small"
        if framing == "aligned":
            h_img, w_img = img.shape[:2]
            ordered      = _img_proc._order_points(quad)
            doc_quad_response = {
                "top_left":     {"x": float(ordered[0][0] / w_img), "y": float(ordered[0][1] / h_img)},
                "top_right":    {"x": float(ordered[1][0] / w_img), "y": float(ordered[1][1] / h_img)},
                "bottom_right": {"x": float(ordered[2][0] / w_img), "y": float(ordered[2][1] / h_img)},
                "bottom_left":  {"x": float(ordered[3][0] / w_img), "y": float(ordered[3][1] / h_img)},
            }

    quality_score = _img_proc._fast_readability_score(img)
    hw = _img_proc.detect_handwriting(img)
    text_type = "handwritten" if hw["is_handwritten"] else "printed"

    effective_threshold = 55
    if hw["is_handwritten"] and hw["confidence"] >= 0.7:
        quality_score = min(quality_score + 8, 100)
        effective_threshold = 50
    elif text_type == "printed":
        effective_threshold = 65

    quality_ok = quality_score >= effective_threshold

    if lighting == "too_dark":
        msg = "Too dark — move to a brighter spot."
    elif lighting == "too_bright":
        msg = "Too bright — tilt slightly to reduce glare."
    elif blur == "blurry":
        msg = "Hold still a moment…"
    elif not quality_ok:
        msg = "Text not clear enough — adjust framing and hold still."
    elif framing == "too_small":
        msg = "Move a little closer to keep the prescription in frame."
    else:
        msg = "Looks good — tap capture!"

    is_usable = (lighting == "good" and blur == "sharp" and quality_ok)
    rejection_reason = (
        None if is_usable
        else "lighting" if lighting != "good"
        else "blur" if blur != "sharp"
        else "readability"
    )

    # ── Compute metrics for DB ─────────────────────────────────────────────────
    _gray_m    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _cy, _cx   = _gray_m.shape[0] // 2, _gray_m.shape[1] // 2
    _roi_m     = _gray_m[int(_cy * 0.3): int(_cy * 1.7), int(_cx * 0.3): int(_cx * 1.7)]
    _mean_v    = float(np.mean(_roi_m))
    _std_v     = float(np.std(_roi_m))
    _blown_pct = float(np.sum(_roi_m > 248)) / _roi_m.size * 100.0
    _dark_pct  = float(np.sum(_roi_m < 30))  / _roi_m.size * 100.0
    _cnr       = _mean_v / _std_v if _std_v > 0.5 else 0.0
    _lap_var   = float(cv2.Laplacian(_gray_m, cv2.CV_64F).var())

    if quad is not None:
        _h_img, _w_img = img.shape[:2]
        _ord         = _img_proc._order_points(quad)
        _db_area_pct = cv2.contourArea(quad) / img_area * 100.0
        _db_detected = True
        _db_tl_x, _db_tl_y = float(_ord[0][0] / _w_img), float(_ord[0][1] / _h_img)
        _db_tr_x, _db_tr_y = float(_ord[1][0] / _w_img), float(_ord[1][1] / _h_img)
        _db_br_x, _db_br_y = float(_ord[2][0] / _w_img), float(_ord[2][1] / _h_img)
        _db_bl_x, _db_bl_y = float(_ord[3][0] / _w_img), float(_ord[3][1] / _h_img)
    else:
        _db_area_pct = 0.0
        _db_detected = False
        _db_tl_x = _db_tl_y = _db_tr_x = _db_tr_y = 0.0
        _db_br_x = _db_br_y = _db_bl_x = _db_bl_y = 0.0

    _elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    if _DB_AVAILABLE:
        try:
            _db_log_validate_capture(
                image_id=image_id,
                quality_score=quality_score,
                blur_score=_lap_var,
                is_usable=is_usable,
                processing_time_ms=_elapsed_ms,
                detection_mode=framing,
                lighting_condition=lighting,
                mean_brightness=_mean_v,
                contrast_ratio=_cnr,
                blown_pixels_percent=_blown_pct,
                dark_pixels_percent=_dark_pct,
                lighting_feedback=msg,
                document_detected=_db_detected,
                top_left_x=_db_tl_x,
                top_left_y=_db_tl_y,
                top_right_x=_db_tr_x,
                top_right_y=_db_tr_y,
                bottom_right_x=_db_br_x,
                bottom_right_y=_db_br_y,
                bottom_left_x=_db_bl_x,
                bottom_left_y=_db_bl_y,
                document_area_percent=_db_area_pct,
                framing_status=framing,
            )
        except Exception as _e:
            print(f"[db] validate-capture log failed: {_e}")

    return {
        "is_usable": is_usable,
        "rejection_reason": rejection_reason,
        "feedback":  {"lighting": lighting, "blur": blur, "framing": framing},
        "instruction": msg,
        "quality_score": quality_score,
        "document_quad": doc_quad_response,
    }


@router.post("/process-image")
async def process_image(
    request: Request,
    file: UploadFile = File(...),
    image_id: str | None = Form(None),
    ground_truth: str | None = Form(None),   # ← QA / CER-only
):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    logger.info(
        "process-image entry image_id=%s request_id=%s",
        image_id, request_id,
    )
    if _DB_AVAILABLE and not image_id:
        raise HTTPException(
            status_code=400,
            detail="image_id is required when database logging is enabled",
        )
    if _DB_AVAILABLE:
        try:
            uuid.UUID(image_id)
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(status_code=400, detail="image_id must be a valid UUID")
        try:
            if not _db_prescription_image_exists(image_id=image_id):
                raise HTTPException(status_code=404, detail="image_id was not found")
        except HTTPException:
            raise
        except Exception as exc:
            print(f"[db] image lookup failed: {exc}")
            raise HTTPException(status_code=503, detail="database lookup failed")

    skip_persistence = False
    if image_id and _DB_AVAILABLE and not skip_persistence:
        try:
            skip_persistence = _db_has_completed_transcription_result(image_id=image_id)
            if skip_persistence:
                logger.info(
                    "persistence skipped: completed result already exists "
                    "(image_id=%s, request_id=%s)",
                    image_id, request_id,
                )
        except Exception as exc:
            logger.error(
                "idempotency check failed (image_id=%s, request_id=%s): %s",
                image_id, request_id, exc,
            )
            raise HTTPException(status_code=503, detail="database idempotency check failed")

    t_start = time.perf_counter()
    preprocessing_started_at = datetime.now(timezone.utc)
    raw     = await file.read()
    if image_id and _DB_AVAILABLE and not skip_persistence:
        try:
            _db_update_image_status(image_id=image_id, status="processing")
        except Exception as exc:
            print(f"[db] image processing status update failed: {exc}")

    img     = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        if image_id and _DB_AVAILABLE and not skip_persistence:
            _db_update_image_status(image_id=image_id, status="failed")
        return {"status": "error", "reason": "invalid_image",
                "message": "Could not decode image."}

    lighting = _img_proc.check_lighting(img)
    if lighting == "good":
        lighting = _img_proc.check_lighting_cnr(img)
    if lighting != "good":
        if image_id and _DB_AVAILABLE and not skip_persistence:
            _db_update_image_status(image_id=image_id, status="rejected")
        return {
            "status": "rejected",
            "reason": lighting,
            "message": "Please adjust lighting and retake.",
        }

    if _img_proc.check_blur(img) == "blurry":
        if image_id and _DB_AVAILABLE:
            _db_update_image_status(image_id=image_id, status="rejected")
        return {
            "status": "rejected",
            "reason": "blurry",
            "message": "Image is too blurry. Hold steady and retake.",
        }

    img, was_corrected = _img_proc.perspective_correct(img)
    detection          = "perspective_corrected" if was_corrected else "frame_region"

    enhanced, readability_confidence, passes_used = _img_proc.enhance_for_handwriting(img)
    preprocessing_id = None
    preprocessing_completed_at = datetime.now(timezone.utc)
    if image_id and _DB_AVAILABLE and not skip_persistence:
        try:
            preprocessing_id = _db_create_preprocessing_log(
                image_id=image_id,
                preprocessing_started_at=preprocessing_started_at,
                preprocessing_completed_at=preprocessing_completed_at,
                total_processing_time_ms=round(
                    (time.perf_counter() - t_start) * 1000, 1
                ),
                target_quality_score=int(readability_confidence),
                pipeline_version=os.getenv("PREPROCESSING_PIPELINE_VERSION", "1"),
                enhancement_passes_used=passes_used,
            )
        except Exception as exc:
            logger.error(
                "preprocessing persistence failed (image_id=%s, step=preprocessing): %s",
                image_id, exc,
            )
    elif _DB_AVAILABLE:
        logger.warning(
            "preprocessing persistence skipped (image_id missing, step=preprocessing)"
        )

    ocr_started = time.perf_counter()
    ocr_provider = "trocr-finetuned"
    try:
        extracted_text, ocr_confidence = run_trocr_ocr(img)
        ocr_error = None
        ocr_failure_reason = None
    except Exception as exc:
        extracted_text = ""
        ocr_confidence = 0.0
        ocr_error = str(exc)
        ocr_failure_reason = str(exc)

    ocr_elapsed_ms = round((time.perf_counter() - ocr_started) * 1000, 1)

    if not extracted_text or len(extracted_text.strip()) < 4:
        if image_id and _DB_AVAILABLE:
            _db_update_image_status(image_id=image_id, status="rejected")
        _, buf_enhanced = cv2.imencode(".png", enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        _, buf_original = cv2.imencode(".jpg", img,      [cv2.IMWRITE_JPEG_QUALITY, 80])

        # ── CER + persist (no_text path) ───────────────────────────────────
        cer_result = None
        if ground_truth:
            cer_result = await asyncio.to_thread(_compute_cer, ground_truth, extracted_text or "")
            gt_check = _check_ground_truth_drug_name(ground_truth)
            cer_result["ground_truth_check"] = gt_check
            print(
                f"[CER] ref_len={cer_result['ref_length']} "
                f"edit_dist={cer_result['edit_distance']} "
                f"CER={cer_result['cer_percent']}% (raw={cer_result['raw_cer_percent']}%, "
                f"gap={cer_result['normalization_gap_percent']}pt) "
                f"PASS={cer_result['passes_objective']}"
            )
            if not skip_persistence:
                _persist_cer(
                    image_id=image_id,
                    ground_truth=ground_truth,
                    hypothesis=extracted_text or "",
                    cer_result=cer_result,
                    request_id=request_id,
                )
        if not skip_persistence:
            try:
                _persist_module_2(
                    image_id=image_id,
                    preprocessing_id=preprocessing_id,
                    provider=ocr_provider,
                    raw_text=extracted_text or "",
                    confidence=ocr_confidence,
                    image_shape=img.shape,
                    medicines=[],
                    raw_medicines=[],
                    inference_time_ms=ocr_elapsed_ms,
                    failure_reason=ocr_failure_reason,
                    request_id=request_id,
                )
            except Exception as exc:
                logger.error("Module 2 persistence failed (image_id=%s, step=no_text): %s", image_id, exc)

        return {
            "status":             "rejected",
            "reason":             "no_text",
            "message":            "No readable text found — move closer or retake.",
            "enhanced_image":     base64.b64encode(buf_enhanced).decode(),
            "original_image":     base64.b64encode(buf_original).decode(),
            "quality_score":      readability_confidence,
            "trocr_confidence":   0,
            "trocr_label":        "No text",
            "ocr_source":         ocr_provider,
            "passes_used":        passes_used,
            "processing_time_ms": round((time.perf_counter() - t_start) * 1000, 1),
            "cer":                cer_result,
        }

    # Structure the recognized text using the prescription parser.
    medicines = (
        parse_prescription(extracted_text, ocr_confidence=ocr_confidence)
        if extracted_text else []
    )
    raw_medicines = [dict(medicine) for medicine in medicines]
    for medicine in medicines:
        for key in (
            "strength", "dosage_form", "dose_amount", "timing",
            "administration_instructions", "duration_source",
        ):
            medicine.setdefault(key, None)
        medicine.pop("_drug_name_raw", None)

    for m in medicines:
        m.pop("dispense_quantity", None)

    for m in medicines:
        dn = m.get("drug_name") or ""
        if dn and not _drug_name_passes_sanity(dn):
            m["drug_name"] = None
            m["uncertain"] = True
        freq = m.get("frequency") or ""
        if dn and freq and dn.lower().strip() == freq.lower().strip():
            m["drug_name"] = None
            m["uncertain"] = True

    ocr_source = ocr_provider

    drug_validations = []
    for m in medicines:
        dn = m.get("drug_name")
        if dn:
            dv = validate_drug_name(dn)
            drug_validations.append(dv)
            if dv["matched"]:
                m["drug_name"] = dv["corrected"]
            else:
                print(f"[main] Discarded unvalidated drug name: {dn!r}")
                m["drug_name"] = None
                m["uncertain"] = True

    first_brand   = medicines[0].get("brand_name") if medicines else None
    first_generic = medicines[0].get("drug_name")  if medicines else None

    if drug_validations:
        drug_validation = drug_validations[0]
    else:
        drug_validation = {"original": "", "corrected": "", "confidence": 0.0, "matched": False}

    ocr_conf_pct = round(ocr_confidence * 100)
    trocr_label  = f"{ocr_conf_pct}%"

    _disp_gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _disp_clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    _disp_img   = _disp_clahe.apply(_disp_gray)
    _, buf_enhanced = cv2.imencode(".png", _disp_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    _, buf_original = cv2.imencode(".jpg", img,       [cv2.IMWRITE_JPEG_QUALITY, 80])

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    if image_id and _DB_AVAILABLE:
        try:
            _db_update_image_status(image_id=image_id, status="processed")
        except Exception as exc:
            print(f"[db] image completion status update failed: {exc}")

    try:
        if skip_persistence:
            logger.info(
                "persistence skipped for final result (image_id=%s, request_id=%s)",
                image_id, request_id,
            )
        else:
            module2_output_id, module2_field_ids = _persist_module_2(
                image_id=image_id,
                preprocessing_id=preprocessing_id,
                provider=ocr_provider,
                raw_text=extracted_text,
                confidence=ocr_confidence,
                image_shape=img.shape,
                medicines=medicines,
                raw_medicines=raw_medicines,
                inference_time_ms=ocr_elapsed_ms,
                failure_reason=ocr_failure_reason,
                request_id=request_id,
            )
            _persist_modules_3_4(
                image_id=image_id,
                output_id=module2_output_id,
                field_ids=module2_field_ids,
                medicines=medicines,
                raw_medicines=raw_medicines,
                ocr_confidence=ocr_confidence,
                request_id=request_id,
            )
    except Exception as exc:
        logger.error("Module 2 persistence failed (image_id=%s, step=success): %s", image_id, exc)

    # ── CER computation + persistence (success path) ──────────────────────────
    cer_result = None
    if ground_truth:
        cer_result = await asyncio.to_thread(_compute_cer, ground_truth, extracted_text)
        gt_check = _check_ground_truth_drug_name(ground_truth)
        cer_result["ground_truth_check"] = gt_check
        print(
            f"[CER] ref_len={cer_result['ref_length']} "
            f"hyp_len={cer_result['hyp_length']} "
            f"S={cer_result['substitutions']} "
            f"D={cer_result['deletions']} "
            f"I={cer_result['insertions']} "
            f"edit_dist={cer_result['edit_distance']} "
            f"CER={cer_result['cer_percent']}% (raw={cer_result['raw_cer_percent']}%, "
            f"gap={cer_result['normalization_gap_percent']}pt) "
            f"PASS={cer_result['passes_objective']}"
        )
        if gt_check and not gt_check["matched"]:
            print(f"[CER] ⚠ ground truth drug name unverified: {gt_check['candidate']!r}")
        if not skip_persistence:
            _persist_cer(
                image_id=image_id,
                ground_truth=ground_truth,
                hypothesis=extracted_text,
                cer_result=cer_result,
                request_id=request_id,
            )

    return {
        "status":             "success",
        "detection":          detection,
        "enhanced_image":     base64.b64encode(buf_enhanced).decode(),
        "original_image":     base64.b64encode(buf_original).decode(),
        "quality_score":      readability_confidence,
        "trocr_confidence":   ocr_conf_pct,
        "trocr_label":        trocr_label,
        "passes_used":        passes_used,
        "processing_time_ms": elapsed_ms,
        "message":            trocr_label,
        "extracted_text":     extracted_text,
        "drug_name":          first_generic or drug_validation["corrected"],
        "brand_name":         first_brand or drug_validation["original"],
        "drug_confidence":    drug_validation["confidence"],
        "drug_matched":       drug_validation["matched"],
        "medicines":          medicines,
        "ocr_source":         ocr_source,
        "cer":                cer_result,
        **({"ocr_error": ocr_error} if ocr_error else {}),
    }


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)