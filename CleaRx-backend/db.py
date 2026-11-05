from __future__ import annotations

import os
import hashlib
from datetime import datetime, timezone
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import execute_values


class DatabaseService:
    @contextmanager
    def _get_conn(self):
        conn = psycopg2.connect(os.getenv("DATABASE_URL"), sslmode="require")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def check_connection(self) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

    def fetch_drug_catalog(self) -> list[tuple[str, str, str, bool]]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, type, COALESCE(is_discontinued, false)
                    FROM drugs
                    WHERE name IS NOT NULL AND btrim(name) <> ''
                    ORDER BY lower(name), COALESCE(is_discontinued, false), id
                    """
                )
                return [
                    (str(drug_id), str(name).strip(), str(drug_type or '').strip(), bool(is_discontinued))
                    for drug_id, name, drug_type, is_discontinued in cur.fetchall()
                ]

    def create_capture_session(
        self,
        *,
        user_id: str,
        device_model: str | None = None,
        os_version: str | None = None,
        app_version: str | None = None,
        image_source: str | None = "camera",
    ) -> str:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO "CAPTURE_SESSION"
                        (user_id, device_model, os_version, app_version,
                         image_source, total_captures, session_status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING session_id
                    """,
                    (
                        user_id, device_model, os_version, app_version,
                        image_source, 0, "active",
                    ),
                )
                return str(cur.fetchone()[0])

    def create_prescription_image(
        self,
        *,
        session_id: str,
        image_bytes: bytes,
        width: int,
        height: int,
        exposure_time: float | None = None,
        iso_sensitivity: int | None = None,
        focal_length: float | None = None,
    ) -> tuple[str, int]:
        checksum = hashlib.sha256(image_bytes).hexdigest()

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM "CAPTURE_SESSION"
                    WHERE session_id = %s
                    FOR UPDATE
                    """,
                    (session_id,),
                )
                if cur.fetchone() is None:
                    raise ValueError(f"capture session {session_id} was not found")

                cur.execute(
                    """
                    SELECT COALESCE(MAX(capture_attempt_number), 0) + 1
                    FROM "PRESCRIPTION_IMAGE"
                    WHERE session_id = %s
                    """,
                    (session_id,),
                )
                attempt_number = int(cur.fetchone()[0])
                cur.execute(
                    """
                    INSERT INTO "PRESCRIPTION_IMAGE"
                        (session_id, file_path, checksum, width, height,
                         exposure_time, iso_sensitivity, focal_length,
                         capture_status, capture_attempt_number)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING image_id
                    """,
                    (
                        session_id, None, checksum, width, height,
                        exposure_time, iso_sensitivity, focal_length,
                        "captured", attempt_number,
                    ),
                )
                image_id = str(cur.fetchone()[0])
                cur.execute(
                    """
                    UPDATE "CAPTURE_SESSION"
                    SET total_captures = total_captures + 1
                    WHERE session_id = %s
                    """,
                    (session_id,),
                )
                if cur.rowcount != 1:
                    raise RuntimeError(
                        f"capture session {session_id} update affected {cur.rowcount} rows"
                    )
                return image_id, attempt_number

    def update_prescription_image_status(self, *, image_id: str, status: str) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE "PRESCRIPTION_IMAGE"
                    SET capture_status = %s
                    WHERE image_id = %s
                    """,
                    (status, image_id),
                )

    def prescription_image_exists(self, *, image_id: str) -> bool:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM "PRESCRIPTION_IMAGE"
                    WHERE image_id = %s
                    """,
                    (image_id,),
                )
                return cur.fetchone() is not None

    def has_completed_transcription_result(self, *, image_id: str) -> bool:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM "PREPROCESSING_LOGS" p
                    JOIN trocr_inference_jobs j
                      ON j.preprocessing_id = p.preprocessing_id
                    JOIN "TRANSCRIPTION_OUTPUTS" o
                      ON o.job_id = j.job_id
                    WHERE p.image_id = %s
                      AND j.job_status = 'completed'
                    LIMIT 1
                    """,
                    (image_id,),
                )
                return cur.fetchone() is not None

    def close_capture_session(self, *, session_id: str, session_status: str) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE "CAPTURE_SESSION"
                    SET ended_at = now(), session_status = %s
                    WHERE session_id = %s
                    """,
                    (session_status, session_id),
                )

    def log_validate_capture(
        self,
        *,
        image_id: str,
        quality_score: int,
        blur_score: float,
        is_usable: bool,
        processing_time_ms: float,
        detection_mode: str,
        lighting_condition: str,
        mean_brightness: float,
        contrast_ratio: float,
        blown_pixels_percent: float,
        dark_pixels_percent: float,
        lighting_feedback: str,
        document_detected: bool,
        top_left_x: float,
        top_left_y: float,
        top_right_x: float,
        top_right_y: float,
        bottom_right_x: float,
        bottom_right_y: float,
        bottom_left_x: float,
        bottom_left_y: float,
        document_area_percent: float,
        framing_status: str,
        validator_version: str | None = None,
        validation_source: str | None = None,
    ) -> str:
        with self._get_conn() as conn:
            cur = conn.cursor()

            cur.execute(
                """
                INSERT INTO "IMAGE_METRICS"
                    (image_id, quality_score, blur_score, is_usable, processing_time_ms,
                     detection_mode, validator_version, validation_source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING metrics_id
                """,
                (image_id, quality_score, blur_score, is_usable, processing_time_ms,
                 detection_mode, validator_version, validation_source),
            )
            metrics_id = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO "LIGHTING_QUALITY"
                    (metrics_id, lighting_condition, mean_brightness, contrast_ratio,
                     blown_pixels_percent, dark_pixels_percent, lighting_feedback)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    metrics_id, lighting_condition, mean_brightness, contrast_ratio,
                    blown_pixels_percent, dark_pixels_percent, lighting_feedback,
                ),
            )

            cur.execute(
                """
                INSERT INTO "FRAMING_QUALITY"
                    (metrics_id, document_detected,
                     top_left_x, top_left_y, top_right_x, top_right_y,
                     bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y,
                     document_area_percent, framing_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    metrics_id, document_detected,
                    top_left_x, top_left_y, top_right_x, top_right_y,
                    bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y,
                    document_area_percent, framing_status,
                ),
            )
            return str(metrics_id)

    def create_preprocessing_log(
        self,
        *,
        image_id: str,
        preprocessing_started_at: datetime,
        preprocessing_completed_at: datetime,
        total_processing_time_ms: float,
        target_quality_score: int,
        pipeline_version: str,
        enhancement_passes_used: int = 0,
    ) -> str:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO "PREPROCESSING_LOGS"
                        (image_id, preprocessing_started_at, preprocessing_completed_at,
                         total_processing_time_ms, target_quality_score,
                         pipeline_version, enhancement_passes_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING preprocessing_id
                    """,
                    (
                        image_id, preprocessing_started_at, preprocessing_completed_at,
                        total_processing_time_ms, target_quality_score,
                        pipeline_version, enhancement_passes_used,
                    ),
                )
                return str(cur.fetchone()[0])

    def get_or_create_trocr_model(
        self,
        *,
        model_name: str,
        model_version: str,
        base_model: str,
        fine_tuned_on: str,
        weights_path: str,
        status: str,
    ) -> str:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trocr_model_registry
                        (model_name, model_version, base_model, fine_tuned_on,
                         weights_path, status)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_name, model_version) DO UPDATE
                        SET base_model = EXCLUDED.base_model,
                            fine_tuned_on = EXCLUDED.fine_tuned_on,
                            weights_path = EXCLUDED.weights_path,
                            status = EXCLUDED.status
                    RETURNING model_id
                    """,
                    (
                        model_name, model_version, base_model, fine_tuned_on,
                        weights_path, status,
                    ),
                )
                return str(cur.fetchone()[0])

    def create_inference_job(
        self,
        *,
        preprocessing_id: str,
        model_id: str,
        input_width_px: int,
        input_height_px: int,
    ) -> str:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trocr_inference_jobs
                        (preprocessing_id, model_id, input_width_px,
                         input_height_px, job_status)
                    VALUES (%s, %s, %s, %s, 'running')
                    RETURNING job_id
                    """,
                    (preprocessing_id, model_id, input_width_px, input_height_px),
                )
                return str(cur.fetchone()[0])

    def finish_inference_job(
        self,
        *,
        job_id: str,
        inference_time_ms: float,
        failure_reason: str | None = None,
    ) -> None:
        status = "failed" if failure_reason else "completed"
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE trocr_inference_jobs
                    SET completed_at = %s, inference_time_ms = %s,
                        job_status = %s, failure_reason = %s
                    WHERE job_id = %s
                    """,
                    (
                        datetime.now(timezone.utc), inference_time_ms, status,
                        failure_reason, job_id,
                    ),
                )
                if cur.rowcount != 1:
                    raise RuntimeError(f"inference job {job_id} was not found")

    def create_transcription_output(
        self,
        *,
        job_id: str,
        raw_text: str,
        overall_confidence: float,
        token_count: int,
        output_status: str,
    ) -> str:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO "TRANSCRIPTION_OUTPUTS"
                        (job_id, raw_text, overall_confidence, token_count, output_status)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING output_id
                    """,
                    (job_id, raw_text, overall_confidence, token_count, output_status),
                )
                return str(cur.fetchone()[0])

    def create_transcribed_fields(
        self,
        *,
        output_id: str,
        fields: list[tuple[str, str, int, float, int, int]],
    ) -> dict[int, str]:
        if not fields:
            return {}
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                rows = execute_values(
                    cur,
                    """
                    INSERT INTO "TRANSCRIBED_FIELDS"
                        (output_id, field_name, field_value, sequence_order,
                         field_confidence, char_start_pos, char_end_pos)
                    VALUES %s
                    RETURNING field_id, sequence_order
                    """,
                    [(output_id, *field) for field in fields],
                    fetch=True,
                )
                return {int(sequence): str(field_id) for field_id, sequence in rows}

    def create_transcription_output_with_fields(
        self,
        *,
        job_id: str,
        raw_text: str,
        overall_confidence: float,
        token_count: int,
        output_status: str,
        fields: list[tuple[str, str, int, float, int, int]],
    ) -> tuple[str, dict[int, str]]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO "TRANSCRIPTION_OUTPUTS"
                        (job_id, raw_text, overall_confidence, token_count, output_status)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING output_id
                    """,
                    (job_id, raw_text, overall_confidence, token_count, output_status),
                )
                output_id = str(cur.fetchone()[0])
                if not fields:
                    return output_id, {}

                rows = execute_values(
                    cur,
                    """
                    INSERT INTO "TRANSCRIBED_FIELDS"
                        (output_id, field_name, field_value, sequence_order,
                         field_confidence, char_start_pos, char_end_pos)
                    VALUES %s
                    RETURNING field_id, sequence_order
                    """,
                    [(output_id, *field) for field in fields],
                    fetch=True,
                )
                return output_id, {
                    int(sequence): str(field_id) for field_id, sequence in rows
                }

    def persist_drug_matches(
        self,
        *,
        matches: list[dict],
    ) -> list[str]:
        if not matches:
            return []
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                match_rows = execute_values(
                    cur,
                    """
                    INSERT INTO drug_match_results
                        (raw_input, matched_name, match_score, field_id, lexicon_id,
                         match_type, is_flagged, flag_reason)
                    VALUES %s
                    RETURNING match_id, field_id
                    """,
                    [
                        (
                            item["raw_input"], item.get("matched_name"),
                            item.get("raw_score"), item["field_id"],
                            item.get("lexicon_id"), item.get("match_type"),
                            item["is_flagged"], item.get("flag_reason"),
                        )
                        for item in matches
                    ],
                    fetch=True,
                )
                match_by_field = {str(field_id): str(match_id) for match_id, field_id in match_rows}
                correction_rows = []
                for item in matches:
                    match_id = match_by_field[str(item["field_id"])]
                    correction_rows.append((
                        match_id, item["original_token"], item["corrected_token"],
                        item["token_sort_ratio"], item["partial_ratio"],
                        item["algorithm_used"], item["score_cutoff_used"],
                        item["correction_applied"],
                    ))
                execute_values(
                    cur,
                    """
                    INSERT INTO ocr_correction_log
                        (match_id, original_token, corrected_token, token_sort_ratio,
                         partial_ratio, algorithm_used, score_cutoff_used,
                         correction_applied)
                    VALUES %s
                    """,
                    correction_rows,
                )
                return [str(match_id) for match_id, _ in match_rows]

    def persist_structured_prescription(
        self,
        *,
        output_id: str,
        prescription_date,
        overall_status: str,
        line_items: list[dict],
    ) -> str | None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        INSERT INTO structured_prescription
                            (output_id, prescription_date, overall_status)
                        VALUES (%s, %s, %s)
                        RETURNING prescription_id
                        """,
                        (output_id, prescription_date, overall_status),
                    )
                    prescription_id = str(cur.fetchone()[0])
                except psycopg2.errors.UniqueViolation:
                    conn.rollback()
                    return None

                item_rows = execute_values(
                    cur,
                    """
                    INSERT INTO prescription_line_items
                        (prescription_id, item_order, drug_name_raw, drug_name_matched,
                         lexicon_id, dosage, frequency, route, duration, quantity,
                         line_item_status, brand_name, strength, dosage_form,
                         dose_amount, timing, meal_relation,
                         administration_instructions, duration_source)
                    VALUES %s
                    RETURNING line_item_id, item_order
                    """,
                    [
                        (
                            prescription_id, item["item_order"], item.get("drug_name_raw"),
                            item.get("drug_name_matched"), item.get("lexicon_id"),
                            item.get("dosage"), item.get("frequency"), item.get("route"),
                            item.get("duration"), item.get("quantity"),
                            item["line_item_status"], item.get("brand_name"),
                            item.get("strength"), item.get("dosage_form"),
                            item.get("dose_amount"), item.get("timing"),
                            item.get("meal_relation"),
                            item.get("administration_instructions"),
                            item.get("duration_source"),
                        )
                        for item in line_items
                    ],
                    fetch=True,
                )
                line_ids = {int(order): str(line_id) for line_id, order in item_rows}
                confidence_rows = []
                for item in line_items:
                    line_id = line_ids[item["item_order"]]
                    for field in item.get("confidence_fields", []):
                        confidence_rows.append((
                            line_id, field["field_name"], field["raw_ocr_confidence"],
                            field.get("fuzzy_match_score"), field["composite_score"],
                            field["confidence_tier"], field["is_flagged"],
                        ))
                if confidence_rows:
                    execute_values(
                        cur,
                        """
                        INSERT INTO field_confidence_scores
                            (line_item_id, field_name, raw_ocr_confidence,
                             fuzzy_match_score, composite_score, confidence_tier,
                             is_flagged)
                        VALUES %s
                        """,
                        confidence_rows,
                    )
                return prescription_id

    # ── NEW: OCR Evaluation (CER) logging ────────────────────────────────────
    def log_ocr_evaluation(
        self,
        *,
        image_id: str,
        ground_truth: str,
        hypothesis: str,
        cer: float,
        cer_percent: float,
        substitutions: int,
        deletions: int,
        insertions: int,
        edit_distance: int,
        ref_length: int,
        hyp_length: int,
        passes_objective: bool,
        target_cer: float = 0.20,
    ) -> str | None:
        """
        Persist one CER evaluation record (Objective 2.2).

        Returns the new evaluation_id (str).
        """
        if not image_id:
            raise ValueError("image_id is required")

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT 1 FROM "OCR_EVALUATION" WHERE image_id = %s LIMIT 1',
                    (image_id,),
                )
                if cur.fetchone() is not None:
                    return None
                cur.execute(
                    """
                    INSERT INTO "OCR_EVALUATION"
                        (image_id, ground_truth, hypothesis,
                         cer, cer_percent,
                         substitutions, deletions, insertions, edit_distance,
                         ref_length, hyp_length,
                         passes_objective, target_cer)
                    VALUES (%s, %s, %s,
                            %s, %s,
                            %s, %s, %s, %s,
                            %s, %s,
                            %s, %s)
                    RETURNING evaluation_id
                    """,
                    (
                        image_id, ground_truth, hypothesis,
                        cer, cer_percent,
                        substitutions, deletions, insertions, edit_distance,
                        ref_length, hyp_length,
                        passes_objective, target_cer,
                    ),
                )
                return str(cur.fetchone()[0])

    def log_drug_validation_evaluation(
        self,
        *,
        image_id: str,
        original_text: str,
        corrected_text: str,
        expected_text: str | None = None,
        match_type: str | None = None,
        confidence: float = 0.0,
        matched: bool = False,
        is_correct_match: bool | None = None,
        is_correctly_flagged: bool | None = None,
        evaluated_at: str | None = None,
    ) -> str | None:
        """Persist one Feature 3 result in the dedicated evaluation table."""
        if not image_id:
            raise ValueError("image_id is required")

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM "DRUG_MATCH_EVALUATION"
                    WHERE image_id = %s AND original_text = %s
                    LIMIT 1
                    """,
                    (image_id, original_text),
                )
                if cur.fetchone() is not None:
                    return None
                cur.execute(
                    """
                    INSERT INTO "DRUG_MATCH_EVALUATION"
                        (image_id, original_text, corrected_text,
                         expected_text, match_type, confidence,
                         matched, is_correct_match, is_correctly_flagged,
                         evaluated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()))
                    RETURNING evaluation_id
                    """,
                    (
                        image_id, original_text, corrected_text,
                        expected_text, match_type, confidence,
                        matched, is_correct_match, is_correctly_flagged,
                        evaluated_at,
                    ),
                )
                return str(cur.fetchone()[0])


# Module-level singleton — preserves the existing import in main.py:
#   from db import log_validate_capture as _db_log_validate_capture
_db_service = DatabaseService()
check_database_connection = _db_service.check_connection
fetch_drug_catalog = _db_service.fetch_drug_catalog
log_validate_capture = _db_service.log_validate_capture
create_capture_session = _db_service.create_capture_session
create_prescription_image = _db_service.create_prescription_image
update_prescription_image_status = _db_service.update_prescription_image_status
prescription_image_exists = _db_service.prescription_image_exists
has_completed_transcription_result = _db_service.has_completed_transcription_result
close_capture_session = _db_service.close_capture_session
log_ocr_evaluation = _db_service.log_ocr_evaluation
log_drug_validation_evaluation = _db_service.log_drug_validation_evaluation
create_preprocessing_log = _db_service.create_preprocessing_log
get_or_create_trocr_model = _db_service.get_or_create_trocr_model
create_inference_job = _db_service.create_inference_job
finish_inference_job = _db_service.finish_inference_job
create_transcription_output = _db_service.create_transcription_output
create_transcribed_fields = _db_service.create_transcribed_fields
create_transcription_output_with_fields = _db_service.create_transcription_output_with_fields
persist_drug_matches = _db_service.persist_drug_matches
persist_structured_prescription = _db_service.persist_structured_prescription