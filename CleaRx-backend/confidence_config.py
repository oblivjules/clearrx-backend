"""Shared informational confidence and status policy.

Scores use 0.0-1.0 except RapidFuzz match scores, which use 0-100.
"""

HIGH_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.60
LOW_OCR_CONFIDENCE = 0.45
LOW_MATCH_CONFIDENCE = 0.60
UNCERTAIN_FIELD_SCORE_CAP = 0.59

STATUS_EXTRACTED = "extracted"
STATUS_EXTRACTED_WITH_FLAGS = "extracted_with_flags"
STATUS_NO_ITEMS = "no_items"

LINE_MATCHED = "matched"
LINE_LOW_CONFIDENCE = "low_confidence"
LINE_UNMATCHED = "unmatched"


def confidence_tier(score: float) -> str:
    if score >= HIGH_CONFIDENCE:
        return "high"
    if score >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"
