-- CleaRx Module 1 capture-quality reports
-- Use :start_time and :end_time as the inclusive/exclusive reporting window.

-- 1. Primary General Objective 1 metric.
-- A usable capture passes lighting, sharpness, document detection, and framing.
WITH attempts AS (
    SELECT DISTINCT
        pi.image_id,
        im.is_usable,
        fq.document_detected,
        fq.framing_status
    FROM "PRESCRIPTION_IMAGE" pi
    JOIN "IMAGE_METRICS" im ON im.image_id = pi.image_id
    LEFT JOIN "FRAMING_QUALITY" fq ON fq.metrics_id = (
        SELECT metrics_id
        FROM "IMAGE_METRICS" im2
        WHERE im2.image_id = pi.image_id
        ORDER BY im2.analyzed_at DESC
        LIMIT 1
    )
    WHERE pi.captured_at >= :start_time
      AND pi.captured_at < :end_time
)
SELECT
    COUNT(*) AS total_capture_attempts,
    COUNT(*) FILTER (WHERE is_usable = true) AS usable_capture_attempts,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE is_usable = true) / NULLIF(COUNT(*), 0),
        2
    ) AS usable_capture_rate_percent,
    CASE
        WHEN COUNT(*) = 0 THEN false
        ELSE COUNT(*) FILTER (WHERE is_usable = true)::numeric / COUNT(*) >= 0.90
    END AS meets_90_percent_target
FROM attempts;

-- 2. Component pass rates.
SELECT
    COUNT(DISTINCT pi.image_id) AS total_capture_attempts,
    ROUND(
        100.0 * COUNT(DISTINCT pi.image_id) FILTER (
            WHERE lq.lighting_condition = 'good'
        ) / NULLIF(COUNT(DISTINCT pi.image_id), 0),
        2
    ) AS lighting_pass_rate_percent,
    ROUND(
        100.0 * COUNT(DISTINCT pi.image_id) FILTER (
            WHERE im.blur_score >= 20.0
        ) / NULLIF(COUNT(DISTINCT pi.image_id), 0),
        2
    ) AS sharpness_pass_rate_percent,
    ROUND(
        100.0 * COUNT(DISTINCT pi.image_id) FILTER (
            WHERE fq.document_detected = true
              AND fq.framing_status = 'aligned'
        ) / NULLIF(COUNT(DISTINCT pi.image_id), 0),
        2
    ) AS framing_pass_rate_percent
FROM "PRESCRIPTION_IMAGE" pi
JOIN "IMAGE_METRICS" im ON im.image_id = pi.image_id
JOIN "LIGHTING_QUALITY" lq ON lq.metrics_id = im.metrics_id
JOIN "FRAMING_QUALITY" fq ON fq.metrics_id = im.metrics_id
WHERE pi.captured_at >= :start_time
  AND pi.captured_at < :end_time;

-- 3. Failure reason distribution.
WITH outcomes AS (
    SELECT DISTINCT ON (pi.image_id)
        pi.image_id,
        CASE
            WHEN im.is_usable = true
             AND fq.document_detected = true
             AND fq.framing_status = 'aligned' THEN 'usable'
            WHEN lq.lighting_condition <> 'good' THEN 'lighting'
            WHEN im.blur_score < 20.0 THEN 'blur'
            WHEN fq.document_detected = false
              OR fq.framing_status <> 'aligned' THEN 'framing'
            ELSE 'other'
        END AS outcome
    FROM "PRESCRIPTION_IMAGE" pi
    JOIN "IMAGE_METRICS" im ON im.image_id = pi.image_id
    JOIN "LIGHTING_QUALITY" lq ON lq.metrics_id = im.metrics_id
    JOIN "FRAMING_QUALITY" fq ON fq.metrics_id = im.metrics_id
    WHERE pi.captured_at >= :start_time
      AND pi.captured_at < :end_time
    ORDER BY pi.image_id, im.analyzed_at DESC
)
SELECT
    outcome,
    COUNT(*) AS capture_count,
    ROUND(100.0 * COUNT(*) / NULLIF(SUM(COUNT(*)) OVER (), 0), 2) AS percentage
FROM outcomes
GROUP BY outcome
ORDER BY capture_count DESC;

-- 4. Session-level retake and completion summary.
SELECT
    session_status,
    COUNT(*) AS session_count,
    SUM(total_captures) AS total_capture_attempts,
    ROUND(AVG(total_captures), 2) AS average_attempts_per_session
FROM "CAPTURE_SESSION"
WHERE started_at >= :start_time
  AND started_at < :end_time
GROUP BY session_status
ORDER BY session_status;
