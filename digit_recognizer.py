"""
digit_recognizer.py - Digit recognition module using normalized template matching.
Independent from operator recognition. Evaluates digits against 0..9 templates
and computes a unified confidence metric.
"""

import os
import cv2
import numpy as np
from segmentation import segment_digits, normalize_roi

DEFAULT_CONFIDENCE_THRESHOLD = 0.70

class DigitRecognizer:
    def __init__(self, templates_dir="templates/digits"):
        self.templates_dir = templates_dir
        self.templates = {}
        self.load_templates()
        
    def load_templates(self):
        """Loads all digit templates 0..9."""
        for d in range(10):
            path = os.path.join(self.templates_dir, f"{d}.png")
            if os.path.exists(path):
                tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                # Ensure binary
                _, tpl_bin = cv2.threshold(tpl, 127, 255, cv2.THRESH_BINARY)
                self.templates[str(d)] = tpl_bin
            else:
                raise FileNotFoundError(f"Template not found: {path}")

    def score_match(self, roi_bin: np.ndarray, template_bin: np.ndarray) -> float:
        """
        Computes a unified similarity score (0.0 to 1.0) combining:
        1. Normalized Cross-Correlation (TM_CCOEFF_NORMED)
        2. Binary Intersection-over-Union (IoU)
        """
        # 1. TM_CCOEFF_NORMED
        res = cv2.matchTemplate(roi_bin, template_bin, cv2.TM_CCOEFF_NORMED)
        corr = float(res[0][0])
        corr_score = max(0.0, corr)
        
        # 2. IoU
        intersection = np.sum((roi_bin > 0) & (template_bin > 0))
        union = np.sum((roi_bin > 0) | (template_bin > 0))
        iou_score = float(intersection) / max(1.0, float(union))
        
        # Unified confidence
        confidence = 0.6 * corr_score + 0.4 * iou_score
        return round(float(confidence), 4)

    def recognize_single_digit(self, norm_roi: np.ndarray) -> dict:
        """Matches a single normalized digit ROI against templates 0..9."""
        best_char = None
        best_conf = -1.0
        scores = {}
        
        for d_str, tpl in self.templates.items():
            conf = self.score_match(norm_roi, tpl)
            scores[d_str] = conf
            if conf > best_conf:
                best_conf = conf
                best_char = d_str
                
        return {
            "char": best_char,
            "confidence": best_conf,
            "all_scores": scores
        }

    def recognize_image(self, binary_mask: np.ndarray, min_conf=DEFAULT_CONFIDENCE_THRESHOLD) -> dict:
        """
        Takes a binary mask (e.g. from preprocessing pipeline),
        segments digits from left to right, and recognizes each digit.
        Assembles multi-digit numbers (e.g., 9 and 2 -> 92).
        """
        segments = segment_digits(binary_mask)
        if not segments:
            return {
                "success": False,
                "value": None,
                "string": "",
                "confidence": 0.0,
                "digits": [],
                "error": "No digits segmented"
            }
            
        recognized_digits = []
        confidences = []
        
        for seg in segments:
            rec = self.recognize_single_digit(seg["normalized_roi"])
            recognized_digits.append({
                "char": rec["char"],
                "confidence": rec["confidence"],
                "bbox": seg["bbox"]
            })
            confidences.append(rec["confidence"])
            
        full_str = "".join([d["char"] for d in recognized_digits])
        
        # Unified overall confidence: weighted average with minimum penalty
        min_c = min(confidences)
        avg_c = sum(confidences) / len(confidences)
        overall_conf = round(0.7 * min_c + 0.3 * avg_c, 4)
        
        success = (overall_conf >= min_conf) and (len(full_str) > 0)
        
        try:
            val = int(full_str)
        except ValueError:
            val = None
            success = False
            
        return {
            "success": success,
            "value": val,
            "string": full_str,
            "confidence": overall_conf,
            "digits": recognized_digits,
            "error": None if success else f"Confidence {overall_conf:.2f} below threshold {min_conf:.2f}"
        }
