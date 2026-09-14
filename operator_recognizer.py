"""
operator_recognizer.py - Operator recognition module (+, -, ×, ÷).
Completely separated from digit recognition.
Combines simple geometric/shape analysis (Connected Components, aspect ratio,
stroke distribution) with normalized template matching for robust confirmation.
"""

import os
import cv2
import numpy as np
from edge_detection import find_connected_components, apply_canny
from segmentation import normalize_roi, DEFAULT_TARGET_SIZE

DEFAULT_CONFIDENCE_THRESHOLD = 0.70

class OperatorRecognizer:
    def __init__(self, templates_dir="templates/operators"):
        self.templates_dir = templates_dir
        self.templates = {}
        self.load_templates()

    def load_templates(self):
        """Loads operator templates: add (+), sub (-), mul (×), div (÷)."""
        mapping = {
            "+": "add.png",
            "-": "sub.png",
            "×": "mul.png",
            "÷": "div.png"
        }
        for op, filename in mapping.items():
            path = os.path.join(self.templates_dir, filename)
            if os.path.exists(path):
                tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                _, tpl_bin = cv2.threshold(tpl, 127, 255, cv2.THRESH_BINARY)
                self.templates[op] = tpl_bin
            else:
                raise FileNotFoundError(f"Template operator not found: {path}")

    def score_template(self, roi_bin: np.ndarray, template_bin: np.ndarray) -> float:
        """Unified template matching score (TM_CCOEFF_NORMED + IoU)."""
        res = cv2.matchTemplate(roi_bin, template_bin, cv2.TM_CCOEFF_NORMED)
        corr_score = max(0.0, float(res[0][0]))
        
        intersection = np.sum((roi_bin > 0) & (template_bin > 0))
        union = np.sum((roi_bin > 0) | (template_bin > 0))
        iou_score = float(intersection) / max(1.0, float(union))
        
        return 0.6 * corr_score + 0.4 * iou_score

    def analyze_geometry(self, components: list[dict], binary_mask: np.ndarray) -> dict:
        """
        Simple geometric shape analysis using Connected Components:
        - 3 components aligned vertically (dot, bar, dot) -> '÷'
        - 1 component with W/H > 2.0 -> '-'
        - 1 component roughly square -> examine cross (+) vs diagonal (×)
        """
        num_comps = len(components)
        
        # 1. Division check: 3 vertically aligned components (or 2 if one dot merged/lost)
        if num_comps == 3:
            # Sort by y-coordinate (top to bottom)
            sorted_by_y = sorted(components, key=lambda c: c["bbox"][1])
            top_comp, mid_comp, bot_comp = sorted_by_y
            
            # Middle component should be horizontal bar (aspect ratio > 1.5)
            mid_ar = mid_comp["aspect_ratio"]
            if mid_ar > 1.4:
                return {"suggested": "÷", "confidence_bonus": 0.25, "reason": "3-component division (dot-bar-dot)"}
        elif num_comps == 2:
            # Sometimes 1 dot + 1 bar
            sorted_by_y = sorted(components, key=lambda c: c["bbox"][1])
            for comp in sorted_by_y:
                if comp["aspect_ratio"] > 1.6:
                    return {"suggested": "÷", "confidence_bonus": 0.15, "reason": "2-component division"}

        # If 1 primary component (or after taking the largest):
        if num_comps >= 1:
            largest = max(components, key=lambda c: c["area"])
            ar = largest["aspect_ratio"]
            
            # Horizontal line check: Minus '-'
            if ar >= 2.0:
                return {"suggested": "-", "confidence_bonus": 0.25, "reason": "High aspect-ratio horizontal stroke (-)"}
            
            # Square-like bounding box: Check + vs ×
            if 0.7 <= ar <= 1.4:
                # Analyze central pixel distribution vs corners
                norm = normalize_roi(largest["crop"], (32, 32))
                h, w = norm.shape
                
                # In '+', middle cross (row 15-17, col 15-17) is filled, but 4 corners are empty
                # In '×', corners have diagonals
                corner_sum = (
                    np.sum(norm[2:8, 2:8] > 0) +
                    np.sum(norm[2:8, 24:30] > 0) +
                    np.sum(norm[24:30, 2:8] > 0) +
                    np.sum(norm[24:30, 24:30] > 0)
                )
                
                # Cross arms at top-center, bottom-center, left-center, right-center
                plus_arms_sum = (
                    np.sum(norm[2:8, 14:18] > 0) +
                    np.sum(norm[24:30, 14:18] > 0) +
                    np.sum(norm[14:18, 2:8] > 0) +
                    np.sum(norm[14:18, 24:30] > 0)
                )
                
                if plus_arms_sum > corner_sum * 1.5:
                    return {"suggested": "+", "confidence_bonus": 0.20, "reason": "Axis-aligned cross (+)"}
                elif corner_sum > plus_arms_sum * 1.2:
                    return {"suggested": "×", "confidence_bonus": 0.20, "reason": "Diagonal strokes (×)"}

        return {"suggested": None, "confidence_bonus": 0.0, "reason": "Generic shape"}

    def recognize_image(self, binary_mask: np.ndarray, min_conf=DEFAULT_CONFIDENCE_THRESHOLD) -> dict:
        """
        Recognizes operator from binary mask (+, -, ×, ÷).
        Integrates Canny edge information, Connected Components, and Template Matching.
        """
        # Extract connected components (ignoring tiny noise < 15px)
        comps = find_connected_components(binary_mask, min_area=15)
        if not comps:
            return {
                "success": False,
                "symbol": None,
                "confidence": 0.0,
                "reason": "No components found",
                "all_scores": {}
            }

        # Step 1: Geometric structural analysis
        geom = self.analyze_geometry(comps, binary_mask)

        # Step 2: Extract unified ROI covering all valid operator components
        min_x = min(c["bbox"][0] for c in comps)
        min_y = min(c["bbox"][1] for c in comps)
        max_x = max(c["bbox"][0] + c["bbox"][2] for c in comps)
        max_y = max(c["bbox"][1] + c["bbox"][3] for c in comps)
        
        overall_crop = binary_mask[min_y:max_y, min_x:max_x]
        norm_roi = normalize_roi(overall_crop, DEFAULT_TARGET_SIZE)

        # Step 3: Template Matching across +, -, ×, ÷
        scores = {}
        for op, tpl in self.templates.items():
            base_score = self.score_template(norm_roi, tpl)
            # Apply geometric bonus if geometry agrees
            if geom["suggested"] == op:
                base_score = min(1.0, base_score + geom["confidence_bonus"])
            scores[op] = round(float(base_score), 4)

        # Pick best match
        best_op = max(scores, key=scores.get)
        best_conf = scores[best_op]
        
        success = (best_conf >= min_conf)
        
        return {
            "success": success,
            "symbol": best_op,
            "confidence": best_conf,
            "bbox": (min_x, min_y, max_x - min_x, max_y - min_y),
            "geometry": geom,
            "all_scores": scores,
            "error": None if success else f"Confidence {best_conf:.2f} below threshold {min_conf:.2f}"
        }
