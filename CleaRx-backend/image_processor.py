from __future__ import annotations

import math
import cv2
import numpy as np


class ImageProcessor:
    """Handles all image quality checks and handwriting enhancement for CleaRx."""

    # ── Lighting / blur thresholds ─────────────────────────────────────────────
    _DARK_THRESHOLD     = 30
    _BRIGHT_THRESHOLD   = 235
    _BLOWN_PCT_LIMIT    = 0.25
    _BLUR_THRESHOLD     = 20.0

    _CNR_TOO_LOW        = 0.8
    _CNR_TOO_HIGH       = 25.0
    _MEAN_ABSOLUTE_DARK = 18.0

    # ── Processing resolutions ─────────────────────────────────────────────────
    _TROCR_TARGET       = 90    # aim for genuine 90% readability
    _WORKING_RESOLUTION = 800   # TrOCR resizes input to 384×384 internally; 800 is sufficient
    _FINAL_RESOLUTION   = 800

    # ── Lighting / blur checks ─────────────────────────────────────────────────

    def check_lighting(self, img: np.ndarray) -> str:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        roi = gray[int(cy * 0.3): int(cy * 1.7), int(cx * 0.3): int(cx * 1.7)]
        mean_val  = float(np.mean(roi))
        blown_pct = float(np.sum(roi > 248)) / roi.size
        if mean_val < self._DARK_THRESHOLD:
            return "too_dark"
        if mean_val > self._BRIGHT_THRESHOLD and blown_pct > self._BLOWN_PCT_LIMIT:
            return "too_bright"
        return "good"

    def check_lighting_cnr(self, img: np.ndarray) -> str:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        roi    = gray[int(cy * 0.3): int(cy * 1.7), int(cx * 0.3): int(cx * 1.7)]
        mean_v = float(np.mean(roi))
        std_v  = float(np.std(roi))
        cnr    = mean_v / std_v if std_v > 0.5 else 0.0
        if mean_v < self._MEAN_ABSOLUTE_DARK or cnr < self._CNR_TOO_LOW:
            return "too_dark"
        if cnr > self._CNR_TOO_HIGH:
            return "too_bright"
        return "good"

    def check_blur(self, img: np.ndarray) -> str:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        roi = gray[int(cy * 0.3): int(cy * 1.7), int(cx * 0.3): int(cx * 1.7)]
        lap_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        return "sharp" if lap_var >= self._BLUR_THRESHOLD else "blurry"

    # ── Perspective correction ─────────────────────────────────────────────────

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect

    def _quad_aspect_ok(self, quad: np.ndarray) -> bool:
        ordered = self._order_points(quad)
        tl, tr, br, bl = ordered
        width  = float(np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        height = float(np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        if width < 80 or height < 80:
            return False
        ratio = height / max(width, 1)
        return 0.25 < ratio < 4.0

    def _best_quad(self, cnts, img_area: float, min_coverage: float):
        for c in cnts:
            if cv2.contourArea(c) < img_area * min_coverage:
                continue
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype("float32")
                if self._quad_aspect_ok(quad):
                    return quad
        return None

    def _best_quad_from_mask(self, mask: np.ndarray, img_area: float, min_coverage: float):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        return self._best_quad(contours, img_area, min_coverage)

    def find_paper_quad(self, img: np.ndarray, min_coverage: float = 0.08):
        h, w  = img.shape[:2]
        scale = 800.0 / max(h, w)
        small = cv2.resize(img, (int(w * scale), int(h * scale)))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        sarea = small.shape[0] * small.shape[1]
        for (lo, hi, iters) in [(50, 150, 1), (20, 100, 2), (10, 60, 3)]:
            edges   = cv2.dilate(cv2.Canny(blur, lo, hi), None, iterations=iters)
            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts    = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
            quad    = self._best_quad(cnts, sarea, min_coverage)
            if quad is not None:
                return quad / scale

        # Low-contrast white paper may have no usable outer Canny contour.
        # A bright-region mask recovers that boundary when the background is darker.
        for threshold in (180, 150, 120):
            mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            quad = self._best_quad_from_mask(mask, sarea, min_coverage)
            if quad is not None:
                return quad / scale
        return None

    def perspective_correct(self, img: np.ndarray) -> tuple[np.ndarray, bool]:
        quad = self.find_paper_quad(img, min_coverage=0.08)
        if quad is None:
            return img, False
        ordered        = self._order_points(quad)
        tl, tr, br, bl = ordered
        width  = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        if width < 100 or height < 100:
            return img, False
        dst = np.array([
            [0,         0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0,         height - 1],
        ], dtype="float32")
        M      = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LANCZOS4)
        return warped, True

    # ── Core image processing primitives ───────────────────────────────────────

    def _estimate_background_fast(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        ksize = max(51, min(h, w) // 40)
        if ksize % 2 == 0:
            ksize += 1
        median_bg = cv2.medianBlur(gray, ksize)
        bg = cv2.GaussianBlur(median_bg, (0, 0), sigmaX=ksize // 3)
        bg = np.clip(bg, 1.0, 255)
        return bg.astype(np.float32)

    def _detect_fold_mask(self, gray: np.ndarray) -> np.ndarray:
        h, w         = gray.shape
        very_blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(5, min(h, w) // 30))
        edges        = cv2.Canny(very_blurred, 15, 45)
        fold_mask    = np.zeros((h, w), dtype=np.uint8)
        min_line_len = int(min(h, w) * 0.35)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=int(min_line_len * 0.8),
            minLineLength=min_line_len, maxLineGap=40,
        )
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = abs(x2 - x1), abs(y2 - y1)
                angle  = np.degrees(np.arctan2(dy, dx))
                if angle < 20 or angle > 70:
                    cv2.line(fold_mask, (x1, y1), (x2, y2), 255,
                             thickness=max(8, min(h, w) // 120))
        dil_k     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fold_mask = cv2.dilate(fold_mask, dil_k, iterations=3)
        return fold_mask

    def _filter_word_components(self, ink_mask: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
        speck_max  = 20
        blob_max   = img_h * img_w * 0.06
        line_ratio = 20.0
        nb, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
        cleaned = np.zeros_like(ink_mask)
        for i in range(1, nb):
            area = stats[i, cv2.CC_STAT_AREA]
            bw   = stats[i, cv2.CC_STAT_WIDTH]
            bh   = stats[i, cv2.CC_STAT_HEIGHT]
            if area < speck_max:
                continue
            if area > blob_max:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > line_ratio:
                continue
            if bh > img_h * 0.80 or bw > img_w * 0.80:
                continue
            cleaned[labels == i] = 255
        return cleaned

    def _sauvola_threshold(self, gray: np.ndarray, win: int = 45, k: float = 0.18) -> np.ndarray:
        gray_f = gray.astype(np.float64)
        mean   = cv2.boxFilter(gray_f,    -1, (win, win), normalize=True)
        sq     = cv2.boxFilter(gray_f**2, -1, (win, win), normalize=True)
        stddev = np.sqrt(np.maximum(sq - mean**2, 0))
        thresh = mean * (1.0 + k * (stddev / 128.0 - 1.0))
        return np.where(gray_f < thresh, 0, 255).astype(np.uint8)

    def _apply_gamma_clahe(self, gray: np.ndarray, gamma: float = 0.85, clip: float = 1.5) -> np.ndarray:
        lut    = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        gammed = cv2.LUT(gray, lut)
        clahe  = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        return clahe.apply(gammed)

    def _adaptive_unsharp_mask(
        self,
        gray: np.ndarray,
        amount: float = 0.45,
        sigma: float = 1.5,
        threshold: int = 5,
    ) -> np.ndarray:
        blurred   = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        diff      = gray.astype(np.float32) - blurred.astype(np.float32)
        edge_mask = (np.abs(diff) > threshold).astype(np.float32)
        sharpened = gray.astype(np.float32) + amount * diff * edge_mask
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _connect_strokes(self, binary_ink: np.ndarray, max_gap: int = 2) -> np.ndarray:
        dist = cv2.distanceTransform(binary_ink, cv2.DIST_L2, 5)
        median_half = np.median(dist[binary_ink > 0]) if np.any(binary_ink) else 1.0
        stroke_w = max(1.0, median_half * 2.0)
        kernel_size = max(2, min(5, int(stroke_w // 2 + 1)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(binary_ink, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.bitwise_or(binary_ink, closed)

    def _sharpen_strokes(self, binary_ink: np.ndarray) -> np.ndarray:
        if np.sum(binary_ink) == 0:
            return binary_ink
        dist = cv2.distanceTransform(binary_ink, cv2.DIST_L2, 5)
        ink_vals = dist[binary_ink > 0]
        if len(ink_vals) < 20:
            return binary_ink
        median_half = float(np.median(ink_vals))
        stroke_w = median_half * 2.0

        if stroke_w < 1.8:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary_ink = cv2.dilate(binary_ink, kernel, iterations=1)
        elif stroke_w > 6.5:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary_ink = cv2.erode(binary_ink, kernel, iterations=1)

        binary_ink = self._connect_strokes(binary_ink, max_gap=2)

        nb, labels, stats, _ = cv2.connectedComponentsWithStats(binary_ink, connectivity=8)
        for i in range(1, nb):
            if stats[i, cv2.CC_STAT_AREA] < 8:
                binary_ink[labels == i] = 0
        return binary_ink

    def _binarise_and_clean(
        self,
        denoised: np.ndarray,
        hh: int,
        ww: int,
        sauvola_k: float,
        dilation_iters: int,
        fold_mask_scaled: np.ndarray | None,
        use_sharpen: bool = True,
        pre_sharpen: bool = True,
    ) -> np.ndarray:
        working = self._adaptive_unsharp_mask(denoised, amount=0.40, sigma=1.5, threshold=4) \
                  if pre_sharpen else denoised

        roi_mean  = float(np.mean(working))
        dynamic_c = int(np.clip(roi_mean * 0.07, 4, 14))

        binary = cv2.adaptiveThreshold(
            working, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=31, C=dynamic_c,
        )
        inv = cv2.bitwise_not(binary)

        sauvola_inv = cv2.bitwise_not(self._sauvola_threshold(working, k=sauvola_k))
        inv = cv2.bitwise_or(inv, sauvola_inv)

        kernel_bold = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        inv = cv2.dilate(inv, kernel_bold, iterations=dilation_iters)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        inv = self._filter_word_components(inv, hh, ww)

        if fold_mask_scaled is not None and np.any(fold_mask_scaled > 0):
            fold_eroded = cv2.erode(
                fold_mask_scaled,
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                iterations=1,
            )
            inv = cv2.bitwise_and(inv, cv2.bitwise_not(fold_eroded))

        if use_sharpen:
            inv = self._sharpen_strokes(inv)

        result = cv2.bitwise_not(inv)

        if fold_mask_scaled is not None and np.any(fold_mask_scaled > 0):
            result[fold_mask_scaled > 0] = 255

        return result

    def _nuclear_pass(
        self,
        gray_normalised: np.ndarray,
        hh: int,
        ww: int,
        fold_mask_scaled: np.ndarray | None,
    ) -> np.ndarray:
        unsharp  = self._adaptive_unsharp_mask(gray_normalised, amount=0.55, sigma=2.0, threshold=5)
        clahe    = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(unsharp)

        gamma = 0.70
        lut   = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        enhanced = cv2.LUT(enhanced, lut)
        enhanced = cv2.bilateralFilter(enhanced, d=7, sigmaColor=30, sigmaSpace=35)

        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = cv2.bitwise_not(otsu)

        sauvola_inv = cv2.bitwise_not(self._sauvola_threshold(enhanced, k=0.22))
        inv = cv2.bitwise_or(inv, sauvola_inv)

        inv = self._connect_strokes(inv)
        inv = self._filter_word_components(inv, hh, ww)

        if fold_mask_scaled is not None and np.any(fold_mask_scaled > 0):
            inv = cv2.bitwise_and(inv, cv2.bitwise_not(fold_mask_scaled))

        inv = self._sharpen_strokes(inv)
        result = cv2.bitwise_not(inv)
        if fold_mask_scaled is not None and np.any(fold_mask_scaled > 0):
            result[fold_mask_scaled > 0] = 255
        return result

    def _finalize(self, result: np.ndarray) -> np.ndarray:
        _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
        bordered = cv2.copyMakeBorder(result, 60, 60, 60, 60,
                                       cv2.BORDER_CONSTANT, value=255)
        scale_f = self._FINAL_RESOLUTION / max(bordered.shape)
        final = cv2.resize(bordered, None, fx=scale_f, fy=scale_f,
                           interpolation=cv2.INTER_LANCZOS4)
        _, final = cv2.threshold(final, 127, 255, cv2.THRESH_BINARY)
        return final

    # ── Readability scoring ────────────────────────────────────────────────────

    def compute_readability_score(self, grayscale: np.ndarray, ink_mask: np.ndarray) -> int:
        h, w = grayscale.shape
        if ink_mask.shape != grayscale.shape:
            ink_mask = cv2.resize(ink_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        ink_pixels = int(np.sum(ink_mask > 0))
        if ink_pixels < 100:
            return 0

        paper_mask   = (ink_mask == 0)
        paper_pixels = int(np.sum(paper_mask))
        if paper_pixels == 0:
            return 0

        mean_ink   = float(np.mean(grayscale[ink_mask > 0]))
        mean_paper = float(np.mean(grayscale[paper_mask]))
        contrast   = mean_paper - mean_ink
        if contrast < 30:
            contrast_norm = max(0.0, contrast / 100.0) * 0.35
        elif contrast < 60:
            contrast_norm = (0.35 + (contrast - 30) / 30.0 * 0.35)
        else:
            contrast_norm = min(0.70 + (contrast - 60) / 40.0 * 0.30, 1.0)
        contrast_score = contrast_norm * 0.35

        lap_var = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
        if lap_var < 20:
            sharpness_norm = 0.10
        elif lap_var < 100:
            sharpness_norm = 0.20 + (lap_var - 20) / 80.0 * 0.60
        else:
            sharpness_norm = min(0.80 + (lap_var - 100) / 300.0 * 0.20, 1.0)
        sharpness_score = sharpness_norm * 0.30

        dist          = cv2.distanceTransform(ink_mask, cv2.DIST_L2, 5)
        ink_dist_vals = dist[ink_mask > 0]
        ink_dist_vals = ink_dist_vals[ink_dist_vals > 0.5]
        if len(ink_dist_vals) > 30:
            median_half  = float(np.median(ink_dist_vals))
            stroke_width = median_half * 2.0
            if stroke_width < 1.0:
                width_norm = 0.10
            elif stroke_width < 1.8:
                width_norm = 0.10 + (stroke_width - 1.0) * 0.50
            elif stroke_width <= 5.5:
                width_norm = 0.60 + (stroke_width - 1.8) / 3.7 * 0.40
            else:
                width_norm = max(0.30, 1.0 - (stroke_width - 5.5) / 10.0)
            std_width  = float(np.std(ink_dist_vals))
            cv_width   = std_width / (median_half + 1e-6)
            if cv_width > 0.7:
                width_norm *= max(0.55, 1.0 - (cv_width - 0.7) / 0.9)
        else:
            width_norm = 0.20
        width_score = float(np.clip(width_norm, 0.0, 1.0)) * 0.15

        nb, _, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
        speckle_pixels = sum(
            stats[i, cv2.CC_STAT_AREA]
            for i in range(1, nb)
            if stats[i, cv2.CC_STAT_AREA] <= 20
        )
        noise_ratio = speckle_pixels / max(ink_pixels, 1)
        noise_score = max(0.0, 1.0 - noise_ratio * 2.0) * 0.08

        total_pixels = h * w
        ink_density  = ink_pixels / total_pixels
        if ink_density < 0.01:
            density_norm = 0.10
        elif ink_density < 0.03:
            density_norm = 0.30 + (ink_density - 0.01) / 0.02 * 0.50
        elif ink_density <= 0.20:
            density_norm = 0.80 + (ink_density - 0.03) / 0.17 * 0.20
        else:
            density_norm = max(0.20, 1.0 - (ink_density - 0.20) / 0.30)
        density_score = float(np.clip(density_norm, 0.0, 1.0)) * 0.05

        if ink_pixels > 50:
            ink_vals        = grayscale[ink_mask > 0].astype(np.float32)
            ink_cv          = float(np.std(ink_vals)) / (float(np.mean(ink_vals)) + 1e-6)
            uniformity_norm = max(0.0, 1.0 - ink_cv / 1.2)
        else:
            uniformity_norm = 0.50
        uniformity_score = uniformity_norm * 0.07

        raw_score = (contrast_score + sharpness_score + width_score +
                     noise_score + density_score + uniformity_score)

        adjusted = 100.0 / (1.0 + math.exp(-14.0 * (raw_score - 0.65)))
        return int(round(min(adjusted, 100.0)))

    # ── Multi-pass handwriting enhancement ────────────────────────────────────

    def enhance_for_handwriting(self, img: np.ndarray) -> tuple[np.ndarray, int, int]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        scale_up = self._WORKING_RESOLUTION / max(h, w)
        if scale_up != 1.0:
            interp = cv2.INTER_LANCZOS4 if scale_up > 1.0 else cv2.INTER_AREA
            gray = cv2.resize(gray,
                              (max(1, round(w * scale_up)), max(1, round(h * scale_up))),
                              interpolation=interp)

        hh, ww = gray.shape

        bg_float         = self._estimate_background_fast(gray)
        fold_mask_raw    = self._detect_fold_mask(gray)
        fold_mask_scaled = fold_mask_raw.astype(np.uint8) if np.any(fold_mask_raw > 0) else None

        bg_float   = np.maximum(bg_float, 1.0)
        normalised = np.clip(gray.astype(np.float32) / bg_float * 255, 0, 255).astype(np.uint8)

        mean_val   = float(np.mean(normalised))
        gamma_val  = 0.75 if mean_val < 100 else 0.85
        lut        = np.array([((i / 255.0) ** gamma_val) * 255 for i in range(256)], dtype=np.uint8)
        norm_gamma = cv2.LUT(normalised, lut)

        denoised = cv2.bilateralFilter(norm_gamma, d=7, sigmaColor=30, sigmaSpace=35)

        def evaluate(binary_img: np.ndarray, pre_gray: np.ndarray) -> int:
            ink_mask = (binary_img < 128).astype(np.uint8)
            return self.compute_readability_score(pre_gray, ink_mask)

        # Pass 1 — mild
        p1 = self._binarise_and_clean(denoised, hh, ww,
                                      sauvola_k=0.16, dilation_iters=1,
                                      fold_mask_scaled=fold_mask_scaled,
                                      use_sharpen=True, pre_sharpen=True)
        c1 = evaluate(p1, denoised)
        if c1 >= self._TROCR_TARGET:
            return self._finalize(p1), c1, 1

        # Pass 2 — moderate
        p2 = self._binarise_and_clean(denoised, hh, ww,
                                      sauvola_k=0.20, dilation_iters=2,
                                      fold_mask_scaled=fold_mask_scaled,
                                      use_sharpen=True, pre_sharpen=True)
        c2 = evaluate(p2, denoised)
        if c2 >= self._TROCR_TARGET:
            return self._finalize(p2), c2, 2

        # Pass 3 — CLAHE + stronger gamma for uneven lighting
        enhanced3 = self._apply_gamma_clahe(normalised, gamma=0.70, clip=2.5)
        denoised3 = cv2.bilateralFilter(enhanced3, d=9, sigmaColor=35, sigmaSpace=40)
        p3 = self._binarise_and_clean(denoised3, hh, ww,
                                      sauvola_k=0.20, dilation_iters=2,
                                      fold_mask_scaled=fold_mask_scaled,
                                      use_sharpen=True, pre_sharpen=True)
        c3 = evaluate(p3, denoised3)
        if c3 >= self._TROCR_TARGET:
            return self._finalize(p3), c3, 3

        # Pass 4 — nuclear
        p4 = self._nuclear_pass(normalised, hh, ww, fold_mask_scaled)
        c4 = evaluate(p4, normalised)

        best = max(c1, c2, c3, c4)
        if best == c4:
            return self._finalize(p4), c4, 4
        if best == c3:
            return self._finalize(p3), c3, 3
        if best == c2:
            return self._finalize(p2), c2, 2
        return self._finalize(p1), c1, 1

    def _fast_readability_score(self, img: np.ndarray) -> int:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = min(1.0, 640.0 / max(h, w))
        if scale < 1.0:
            gray = cv2.resize(gray,
                              (max(1, int(w * scale)), max(1, int(h * scale))),
                              interpolation=cv2.INTER_AREA)
        bg = self._estimate_background_fast(gray)
        bg = np.maximum(bg, 1.0)
        normalised = np.clip(gray.astype(np.float32) / bg * 255, 0, 255).astype(np.uint8)
        clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced   = clahe.apply(normalised)
        denoised   = cv2.bilateralFilter(enhanced, d=5, sigmaColor=25, sigmaSpace=25)
        binary     = self._sauvola_threshold(denoised, win=31, k=0.16)
        inv        = cv2.bitwise_not(binary)
        inv        = self._filter_word_components(inv, *inv.shape[:2])
        result     = cv2.bitwise_not(inv)
        ink_mask   = (result < 128).astype(np.uint8)
        return self.compute_readability_score(denoised, ink_mask)

    # ── Handwriting type detection ─────────────────────────────────────────────

    def detect_handwriting(self, img: np.ndarray) -> dict:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        scale = min(1.0, 800.0 / max(h, w))
        if scale < 1.0:
            gray = cv2.resize(gray,
                              (max(1, int(w * scale)), max(1, int(h * scale))),
                              interpolation=cv2.INTER_AREA)

        bg         = self._estimate_background_fast(gray)
        bg         = np.maximum(bg, 1.0)
        normalised = np.clip(gray.astype(np.float32) / bg * 255, 0, 255).astype(np.uint8)
        denoised   = cv2.bilateralFilter(normalised, d=5, sigmaColor=25, sigmaSpace=25)
        binary     = self._sauvola_threshold(denoised, win=31, k=0.18)
        ink        = cv2.bitwise_not(binary)

        ink_pixels = int(np.sum(ink > 0))
        if ink_pixels < 100:
            return {"is_handwritten": False, "confidence": 0.0,
                    "stroke_width_cv": 0.0, "irregularity": 0.0}

        dist          = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
        ink_dist_vals = dist[ink > 0]
        ink_dist_vals = ink_dist_vals[ink_dist_vals > 0.5]
        if ink_dist_vals.size < 20:
            sw_cv = 0.0
        else:
            sw_mean = float(np.mean(ink_dist_vals))
            sw_std  = float(np.std(ink_dist_vals))
            sw_cv   = sw_std / sw_mean if sw_mean > 0 else 0.0
        sw_score = float(np.clip((sw_cv - 0.35) / 0.40, 0.0, 1.0))

        try:
            skel   = np.zeros_like(ink)
            temp   = ink.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            for _ in range(25):
                eroded = cv2.erode(temp, kernel)
                opened = cv2.dilate(eroded, kernel)
                skel   = cv2.bitwise_or(skel, cv2.subtract(temp, opened))
                temp   = eroded
                if cv2.countNonZero(temp) == 0:
                    break
            skel_pixels = int(np.sum(skel > 0))
            if skel_pixels < 20:
                irregularity = 0.0
            else:
                skel_u8       = (skel > 0).astype(np.uint8)
                neighbour_sum = cv2.filter2D(skel_u8.astype(np.float32), -1,
                                             np.ones((3, 3), np.float32))
                branch_pts    = int(np.sum((skel_u8 > 0) & (neighbour_sum >= 4)))
                irregularity  = branch_pts / skel_pixels
        except Exception:
            irregularity = 0.0

        irreg_score = float(np.clip((irregularity - 0.20) / 0.30, 0.0, 1.0))
        combined    = 0.60 * sw_score + 0.40 * irreg_score
        is_hw       = combined >= 0.55

        return {
            "is_handwritten": is_hw,
            "confidence": round(combined, 3),
            "stroke_width_cv": round(sw_cv, 3),
            "irregularity": round(irregularity, 3),
        }
