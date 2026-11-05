from __future__ import annotations

import math
import re
import threading
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pathlib import Path


class OCRService:
    """Manages TrOCR model lifecycle and prescription image OCR inference."""

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _BASE_MODEL_ID = "microsoft/trocr-base-handwritten"

    _FT_LOW_CONF_THRESHOLD = 0.25

    # Catches 4+ consecutive identical chars and 3+ two-char alternations
    _HALLUCINATION_RE     = re.compile(r'(.)\1{3,}')
    _HALLUCINATION_ALT_RE = re.compile(r'(..)\1{2,}')

    # Lines below this confidence are too noisy to trust
    _MIN_LINE_CONF = 0.12

    # Recognises a drug-name token followed by a dosage — used for confidence boosting
    _PHARMA_DOSAGE_RE = re.compile(
        r'\b[A-Za-z]{3,}\s+\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|iu|meq)\b',
        re.IGNORECASE,
    )

    # Detects lines composed entirely of frequency/instruction tokens
    _INSTRUCTION_ONLY_RE = re.compile(
        r'^[\s\d]*(?:once|twice|thrice|daily|weekly|nightly|bedtime|every|stat|prn|'
        r'bid|tid|qid|bd|tds|qds|od|qd|hs|qhs|mane|nocte|take|apply|'
        r'tablet[s]?|tab[s]?|capsule[s]?|cap[s]?|'
        r'times?\s+(?:a\s+)?(?:day|daily)|x\s*\d+|q\d+h'
        r')[\s\d.,/]*$',
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        # Locate fine-tuned model checkpoint
        _runs_dir = Path(__file__).resolve().parent / "runs" / "trocr-rx"
        _best = _runs_dir / "best"
        _ep20 = _runs_dir / "checkpoint-epoch-20"
        self._ft_path = _best if _best.exists() else (_ep20 if _ep20.exists() else None)

        # Model instances — fine-tuned is eager, base model is lazy
        self._ft_processor: TrOCRProcessor | None = None
        self._ft_model: VisionEncoderDecoderModel | None = None
        self._processor: TrOCRProcessor | None = None
        self._model: VisionEncoderDecoderModel | None = None
        self._model_load_lock = threading.Lock()

        # Load local models on first inference to keep API startup fast.

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_ft_model(self) -> None:
        if self._ft_path is None or (self._ft_processor is not None and self._ft_model is not None):
            return
        with self._model_load_lock:
            if self._ft_path is None or (self._ft_processor is not None and self._ft_model is not None):
                return
            print(f"[OCR] Loading fine-tuned model: {self._ft_path}")
            self._ft_processor = TrOCRProcessor.from_pretrained(str(self._ft_path))
            self._ft_model = (
                VisionEncoderDecoderModel.from_pretrained(str(self._ft_path))
                .to(self._DEVICE)
            )
            self._ft_model.eval()
            print(f"[OCR] Fine-tuned model loaded on {self._DEVICE}")

    def _load_base_model(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        with self._model_load_lock:
            if self._processor is not None and self._model is not None:
                return
            print(f"[OCR] Loading base model: {self._BASE_MODEL_ID}")
            self._processor = TrOCRProcessor.from_pretrained(self._BASE_MODEL_ID)
            self._model = (
                VisionEncoderDecoderModel.from_pretrained(self._BASE_MODEL_ID)
                .to(self._DEVICE)
            )
            self._model.eval()
            print(f"[OCR] Base model loaded on {self._DEVICE}")

    # ── Image segmentation ─────────────────────────────────────────────────────

    def segment_text_lines(
        self,
        binary_img: np.ndarray,
        min_line_height: int = None,
        gap_threshold: int = 15,
        padding: int = 12,
    ) -> list[np.ndarray]:
        """
        Split a grayscale image (white background, dark ink) into text-line strips
        using horizontal projection profiles with gap-bridging.

        Gaps of up to gap_threshold consecutive empty rows are bridged before a line
        segment is closed, so broken strokes (dots on 'i', ascenders, etc.) do not
        cause over-segmentation without needing morphological dilation.

        Returns the original image as a single element when no clear lines are found.
        """
        h, w = binary_img.shape[:2]

        margin = max(1, int(w * 0.03))
        center = binary_img[:, margin: w - margin]

        _, otsu_bin = cv2.threshold(center, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = (otsu_bin == 0).astype(np.uint8)
        row_has_ink = ink.any(axis=1)

        if min_line_height is None:
            min_line_height = max(12, int(h * 0.03))

        lines: list[np.ndarray] = []
        in_line    = False
        line_start = 0
        gap_count  = 0

        for r in range(h):
            if row_has_ink[r]:
                if not in_line:
                    in_line    = True
                    line_start = r
                gap_count = 0
            else:
                if in_line:
                    gap_count += 1
                    if gap_count > gap_threshold:
                        in_line = False
                        end_row = r - gap_count
                        if end_row - line_start >= min_line_height:
                            top = max(0, line_start - padding)
                            bot = min(h, end_row + padding)
                            lines.append(binary_img[top:bot, :])

        if in_line:
            end_row = h - gap_count
            if end_row - line_start >= min_line_height:
                top = max(0, line_start - padding)
                lines.append(binary_img[top:h, :])

        min_ink_px = max(40, w * 3)
        lines = [ln for ln in lines if int(np.sum(ln < 128)) >= min_ink_px]

        return lines if lines else [binary_img]

    # ── Text cleaning / correction ─────────────────────────────────────────────

    def _clean_trocr_text(self, text: str) -> str:
        """Remove common TrOCR hallucination artifacts from decoded handwriting output."""
        text = self._HALLUCINATION_RE.sub(r'\1', text)
        text = self._HALLUCINATION_ALT_RE.sub(r'\1', text)

        for src, dst in (
            ('‘', "'"), ('’', "'"),
            ('“', '"'), ('”', '"'),
            ('–', '-'), ('—', '-'),
            ('µ', 'u'),
            ('×', 'x'),
            ('°', ''),
            ('•', ''),
            ('．', '.'), ('，', ','),
            (' ', ' '),
        ):
            text = text.replace(src, dst)

        text = re.sub(r'^[^\w\d(]+', '', text)
        text = re.sub(r'[^\w\d)%]+$', '', text)
        text = re.sub(r'  +', ' ', text)
        return text.strip()

    def _pharma_post_correct(self, text: str) -> str:
        """
        Pharmaceutical-domain post-OCR correction for systematic TrOCR errors.
        Only high-confidence, low-false-positive patterns are applied.
        """
        text = re.sub(r'(?<=\d)\s*rng\b',  'mg',  text, flags=re.IGNORECASE)
        text = re.sub(r'(?<=\d)\s*rnl\b',  'ml',  text, flags=re.IGNORECASE)
        text = re.sub(r'(?<=\d)\s*rncg\b', 'mcg', text, flags=re.IGNORECASE)

        text = re.sub(r'\b9([Ii1][Dd])\b', lambda m: 'q' + m.group(1).lower(), text)
        text = re.sub(r'\b9([Hh][Ss])\b',  lambda m: 'q' + m.group(1).lower(), text)
        text = re.sub(r'\b9([Dd])\b(?!\d)', lambda m: 'q' + m.group(1).lower(), text)

        text = re.sub(
            r'(\d)\s(\d{1,2})\s*(mg|mcg|ml|g|iu|meq)\b',
            r'\1\2\3', text, flags=re.IGNORECASE,
        )

        text = re.sub(r'\bmgs\b',   'mg',   text, flags=re.IGNORECASE)
        text = re.sub(r'\btabls?\b', 'tabs', text, flags=re.IGNORECASE)

        text = re.sub(
            r'([A-Za-z]{4,})(\d+(?:\.\d+)?)(mg|mcg|ml|iu|meq)\b',
            r'\1 \2\3', text, flags=re.IGNORECASE,
        )

        text = re.sub(r'\bq\s+(\d+)\s*h(?:rs?)?\b', r'q\1h', text, flags=re.IGNORECASE)
        text = re.sub(r'\bq\.(\d+)\.h\.?(?=\s|[,;)]|\Z)', r'q\1h', text, flags=re.IGNORECASE)
        text = re.sub(r'(?<=\d)\s*lU\b', ' IU', text)
        text = re.sub(r'(?<=\d)(mg|mcg|ml|iu|meq)\.(?=\s|$)', r'\1', text, flags=re.IGNORECASE)

        return text.strip()

    def _is_instruction_only(self, text: str) -> bool:
        """Return True when text is composed entirely of frequency/instruction tokens."""
        cleaned = re.sub(r'\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|ml|g|iu|meq|%)\b', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'[^A-Za-z\s]', ' ', cleaned).strip()
        if not cleaned:
            return True
        return bool(self._INSTRUCTION_ONLY_RE.match(text.strip()))

    def _pharma_confidence_boost(self, text: str, conf: float) -> float:
        """
        Small, capped confidence boost when decoded text contains a recognisable
        drug-name + dosage pattern. Suppressed for instruction-only lines.
        """
        if self._is_instruction_only(text):
            return conf
        if conf < 0.92 and self._PHARMA_DOSAGE_RE.search(text):
            return min(0.92, conf + 0.08)
        return conf

    def _is_noise_line(self, text: str) -> bool:
        """Return True for lines that carry no prescription-relevant content."""
        s = text.strip()
        if len(s) < 3:
            return True
        if not re.search(r'[A-Za-z]', s):
            return True
        if re.match(r'^[\d\s\-\.\/()+]+$', s):
            return True
        alpha = [c.lower() for c in s if c.isalpha()]
        if len(alpha) >= 6 and len(set(alpha)) / len(alpha) < 0.25:
            return True
        return False

    # ── Inference ──────────────────────────────────────────────────────────────

    def _infer_line(
        self,
        pil_img: Image.Image,
        proc: TrOCRProcessor,
        mdl: VisionEncoderDecoderModel,
    ) -> tuple[str, float]:
        """
        Run TrOCR on a single-line PIL image.
        Returns (decoded_text, confidence) where confidence is derived from the
        beam-search sequence log-probability (0.0–1.0).
        """
        pixel_values = proc(images=pil_img, return_tensors="pt").pixel_values.to(self._DEVICE)

        with torch.no_grad():
            outputs = mdl.generate(
                pixel_values,
                max_new_tokens=96,
                num_beams=8,
                early_stopping=True,
                length_penalty=0.9,
                repetition_penalty=1.4,
                no_repeat_ngram_size=3,
                return_dict_in_generate=True,
                output_scores=True,
            )

        text = proc.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
        text = self._clean_trocr_text(text)
        text = self._pharma_post_correct(text)

        # Centre the sigmoid at -1.0 (reasonable mid-confidence for TrOCR on Rx text)
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            score = float(outputs.sequences_scores[0])
            confidence = round(1.0 / (1.0 + math.exp(-(score + 1.0) * 3.0)), 3)
        else:
            confidence = 0.5

        return text, confidence

    # ── Image pre-processing helpers ───────────────────────────────────────────

    def _sharpen_line(self, crop: np.ndarray) -> np.ndarray:
        """Light unsharp masking to enhance edges before OCR."""
        blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.0)
        sharp = cv2.addWeighted(crop, 1.5, blurred, -0.5, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    def _sharpen_blurry_crop(self, crop: np.ndarray) -> np.ndarray:
        """Stronger unsharp masking applied only when Laplacian variance is low."""
        lap_var = float(cv2.Laplacian(crop.astype(np.float64), cv2.CV_64F).var())
        if lap_var < 25.0:
            blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.5)
            sharp = cv2.addWeighted(crop, 2.0, blurred, -1.0, 0)
            return np.clip(sharp, 0, 255).astype(np.uint8)
        return crop

    def _correct_skew(self, gray: np.ndarray, max_angle: float = 12.0) -> np.ndarray:
        """
        Detect and correct document skew using Hough line voting on Canny edges.
        Only acts when skew is between 1° and max_angle (12°).
        """
        h, w = gray.shape
        scale = min(1.0, 800.0 / max(h, w))
        if scale < 1.0:
            small = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = gray.copy()

        edges = cv2.Canny(small, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=60,
            minLineLength=max(30, small.shape[1] // 5),
            maxLineGap=15,
        )
        if lines is None:
            return gray

        angles: list[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(ang) <= max_angle:
                angles.append(ang)

        if not angles:
            return gray

        median_angle = float(np.median(angles))
        if abs(median_angle) < 1.0:
            return gray

        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), median_angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def _decircle(self, gray: np.ndarray) -> np.ndarray:
        """
        Erase hand-drawn oval/circle annotations common in Philippine prescriptions.
        Currently disabled in run_trocr — was erasing circled digits (doses/quantities).
        """
        h, w = gray.shape
        _, ink = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = gray.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 150:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri < 1:
                continue
            circularity = 4.0 * math.pi * area / (peri ** 2)
            if circularity < 0.35:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            asp = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
            if asp < 0.25:
                continue
            if not (50 < bw < w * 0.45 and 30 < bh < h * 0.45):
                continue
            thickness = max(3, int(min(bw, bh) * 0.08))
            cv2.drawContours(out, [cnt], -1, 255, thickness * 2)
        return out

    # ── Public OCR entry point ─────────────────────────────────────────────────

    def run_trocr(self, image: np.ndarray) -> tuple[str, float]:
        """
        Accept a BGR (preferred) or grayscale ndarray and return (text, confidence).

        Uses the local fine-tuned checkpoint with a base TrOCR fallback.
        """
        if self._ft_path is not None:
            self._load_ft_model()
        else:
            self._load_base_model()

        if image.ndim == 2:
            gray = image.copy()
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        scale = min(2.0, 1600.0 / max(h, w))
        if abs(scale - 1.0) > 0.05:
            interp = cv2.INTER_LANCZOS4 if scale > 1.0 else cv2.INTER_AREA
            gray = cv2.resize(gray,
                              (max(1, round(w * scale)), max(1, round(h * scale))),
                              interpolation=interp)

        ksize = max(31, min(gray.shape) // 10)
        if ksize % 2 == 0:
            ksize += 1
        ksize = min(ksize, 101)
        bg = cv2.medianBlur(gray, ksize).astype(np.float32)
        bg = np.maximum(bg, 1.0)
        normalized = np.clip(gray.astype(np.float32) / bg * 170.0, 0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        normalized = clahe.apply(normalized)

        denoised = cv2.bilateralFilter(normalized, d=7, sigmaColor=25, sigmaSpace=25)

        # _decircle disabled — was erasing circled digits (doses/quantities)
        # denoised = self._decircle(denoised)

        denoised = self._correct_skew(denoised)

        denoised = cv2.copyMakeBorder(denoised, 30, 30, 30, 30,
                                      cv2.BORDER_CONSTANT, value=255)

        line_crops = self.segment_text_lines(denoised, gap_threshold=8)
        print(f"[DEBUG] segment_text_lines found {len(line_crops)} line(s)")

        texts: list[str] = []
        confidences: list[float] = []

        for crop in line_crops:
            crop_ink = int(np.sum(crop < 128))
            if crop_ink < 50:
                continue

            if len(crop.shape) == 2:
                crop = self._sharpen_blurry_crop(crop)
                crop = self._sharpen_line(crop)
            pil_img = Image.fromarray(crop).convert("RGB")

            if self._ft_path is not None:
                text, conf = self._infer_line(pil_img, self._ft_processor, self._ft_model)
                conf = self._pharma_confidence_boost(text, conf)
                if conf < self._FT_LOW_CONF_THRESHOLD:
                    self._load_base_model()
                    base_text, base_conf = self._infer_line(pil_img, self._processor, self._model)
                    base_conf = self._pharma_confidence_boost(base_text, base_conf)
                    if base_conf > conf:
                        text, conf = base_text, base_conf
            else:
                text, conf = self._infer_line(pil_img, self._processor, self._model)
                conf = self._pharma_confidence_boost(text, conf)

            kept = text and conf >= self._MIN_LINE_CONF and not self._is_noise_line(text)
            print(f"[DEBUG] Line text: {text!r}  conf={conf:.3f}  kept={kept}")
            if kept:
                texts.append(text)
                confidences.append(conf)

        # ── Two-pass fallback ──────────────────────────────────────────────────
        # When first-pass average confidence is below 0.38, re-process with
        # stronger CLAHE and heavier bilateral filtering, then accept whichever
        # pass gives higher average confidence.
        if confidences and (sum(confidences) / len(confidences)) < 0.38:
            clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            boosted = clahe2.apply(normalized)
            boosted = cv2.bilateralFilter(boosted, d=9, sigmaColor=35, sigmaSpace=35)
            boosted = self._correct_skew(boosted)
            boosted = cv2.copyMakeBorder(boosted, 30, 30, 30, 30,
                                         cv2.BORDER_CONSTANT, value=255)
            line_crops2 = self.segment_text_lines(boosted, gap_threshold=10)

            texts2: list[str] = []
            confidences2: list[float] = []
            for crop2 in line_crops2:
                if int(np.sum(crop2 < 128)) < 50:
                    continue
                if len(crop2.shape) == 2:
                    crop2 = self._sharpen_blurry_crop(crop2)
                    crop2 = self._sharpen_line(crop2)
                pil2 = Image.fromarray(crop2).convert("RGB")
                if self._ft_path is not None:
                    t2, c2 = self._infer_line(pil2, self._ft_processor, self._ft_model)
                    c2 = self._pharma_confidence_boost(t2, c2)
                    if c2 < self._FT_LOW_CONF_THRESHOLD:
                        self._load_base_model()
                        bt2, bc2 = self._infer_line(pil2, self._processor, self._model)
                        bc2 = self._pharma_confidence_boost(bt2, bc2)
                        if bc2 > c2:
                            t2, c2 = bt2, bc2
                else:
                    t2, c2 = self._infer_line(pil2, self._processor, self._model)
                    c2 = self._pharma_confidence_boost(t2, c2)
                if t2 and c2 >= self._MIN_LINE_CONF and not self._is_noise_line(t2):
                    texts2.append(t2)
                    confidences2.append(c2)

            if confidences2 and (
                sum(confidences2) / len(confidences2)
                > sum(confidences) / len(confidences)
            ):
                texts = texts2
                confidences = confidences2

        combined_text  = "\n".join(texts)
        avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

        return combined_text, avg_confidence


# Module-level singleton — preserves existing import in main.py:
#   from ocr_service import run_trocr
_ocr_service = OCRService()
run_trocr = _ocr_service.run_trocr
