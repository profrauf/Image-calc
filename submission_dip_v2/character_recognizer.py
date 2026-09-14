"""
character_recognizer.py - High-accuracy Digit Segmenter & Structural Operator Classifier.
Part of DIP-Lab-6 Image-Based Calculator (v2 Architecture).

Features:
1. DigitSegmenterAndRecognizer:
   - Left-to-Right connected components extraction.
   - Aspect-ratio preserved isotropic scaling and centering (36x36 canvas).
   - Multi-digit assembly (e.g., '9' + '5' -> 95) for numbers < 100.
   - Dual-metric matching: Normalized Cross-Correlation (NCC) + Intersection-over-Union (IoU).

2. OperatorClassifier:
   - Structural geometric characterization:
     * Division (÷): 3 vertically collinear components (top dot + bar + bottom dot) or 2 components with a central bar.
     * Subtraction (-): Single high-aspect-ratio horizontal stroke (W/H >= 2.0).
     * Addition (+): Axis-aligned cross; center cross arms filled, corners empty.
     * Multiplication (×): Diagonal cross; diagonals active, axis arms empty.
   - Normalized template matching confirmation.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import cv2
import numpy as np
from dip_engine import DIPEngine

TARGET_DIM = (36, 36)  # Height, Width for normalized templates and ROIs

def normalize_character_roi(roi_binary: np.ndarray, target_size: Tuple[int, int] = TARGET_DIM, margin: int = 4) -> np.ndarray:
    """
    Normalizes a binary foreground mask (255=character, 0=background) into a fixed canvas
    while preserving original aspect ratio and centering the glyph.
    """
    th, tw = target_size
    pad_canvas = np.zeros((th, tw), dtype=np.uint8)
    
    coords = cv2.findNonZero(roi_binary)
    if coords is None:
        return pad_canvas
        
    x, y, w, h = cv2.boundingRect(coords)
    if w <= 0 or h <= 0:
        return pad_canvas
        
    crop = roi_binary[y:y+h, x:x+w]
    max_side = max(w, h)
    scale = float(min(th, tw) - (margin * 2)) / float(max_side)
    
    scaled_w = max(1, int(w * scale))
    scaled_h = max(1, int(h * scale))
    
    resized = cv2.resize(crop, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    _, binary_resized = cv2.threshold(resized, 100, 255, cv2.THRESH_BINARY)
    
    start_x = (tw - scaled_w) // 2
    start_y = (th - scaled_h) // 2
    pad_canvas[start_y:start_y+scaled_h, start_x:start_x+scaled_w] = binary_resized
    return pad_canvas


class DigitSegmenterAndRecognizer:
    """
    Segments connected components, sorts them left-to-right, and recognizes digits 0..9.
    """
    
    def __init__(self, templates_dir: str = "templates/digits"):
        self.templates_dir = templates_dir
        self.templates: Dict[str, np.ndarray] = {}
        self.load_templates()

    def load_templates(self) -> None:
        """Loads normalized 0..9 digit templates."""
        for d in range(10):
            d_str = str(d)
            fpath = os.path.join(self.templates_dir, f"{d_str}.png")
            if os.path.exists(fpath):
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                _, binary_tpl = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                self.templates[d_str] = binary_tpl
            else:
                # Placeholder until generated
                self.templates[d_str] = np.zeros(TARGET_DIM, dtype=np.uint8)

    @staticmethod
    def compute_similarity(roi: np.ndarray, tpl: np.ndarray) -> float:
        """
        Computes composite similarity score:
        Composite = 0.6 * NCC + 0.4 * IoU
        """
        # 1. Normalized Cross-Correlation
        match_map = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        ncc_score = max(0.0, float(match_map[0][0]))
        
        # 2. Intersection over Union (IoU)
        intersection = np.sum((roi > 0) & (tpl > 0))
        union = np.sum((roi > 0) | (tpl > 0))
        iou_score = float(intersection) / max(1.0, float(union))
        
        composite = 0.6 * ncc_score + 0.4 * iou_score
        return round(composite, 4)

    def classify_single_digit(self, norm_roi: np.ndarray) -> Dict[str, Any]:
        """Matches a single normalized digit ROI against all 0..9 templates."""
        best_digit = None
        highest_score = -1.0
        all_scores = {}
        
        for d_str, tpl in self.templates.items():
            score = self.compute_similarity(norm_roi, tpl)
            all_scores[d_str] = score
            if score > highest_score:
                highest_score = score
                best_digit = d_str
                
        return {
            "digit": best_digit,
            "confidence": highest_score,
            "scores": all_scores
        }

    def segment_digits(self, binary_mask: np.ndarray, min_area: int = 30, min_height: int = 12) -> List[Dict[str, Any]]:
        """
        Extracts digit components and sorts them strictly Left -> Right.
        """
        comps = DIPEngine.extract_connected_components(binary_mask, min_pixel_area=min_area)
        
        valid_digits = []
        for c in comps:
            x, y, w, h = c["bbox"]
            if h < min_height or w < 3:
                continue
                
            norm_roi = normalize_character_roi(c["crop"], TARGET_DIM)
            valid_digits.append({
                "bbox": (x, y, w, h),
                "area": c["area"],
                "aspect_ratio": c["aspect_ratio"],
                "raw_crop": c["crop"],
                "normalized_roi": norm_roi
            })
            
        # Sort Left-to-Right by x position
        valid_digits.sort(key=lambda item: item["bbox"][0])
        return valid_digits

    def recognize_number(self, binary_mask: np.ndarray, min_confidence: float = 0.70) -> Dict[str, Any]:
        """
        End-to-end number recognition pipeline.
        Segments digits, recognizes each, and constructs full multi-digit number (< 100).
        """
        segments = self.segment_digits(binary_mask)
        if not segments:
            return {
                "success": False,
                "value": None,
                "number_str": "",
                "confidence": 0.0,
                "digits": [],
                "error": "No digit segments detected."
            }

        digit_results = []
        conf_list = []
        
        for seg in segments:
            res = self.classify_single_digit(seg["normalized_roi"])
            digit_results.append({
                "char": res["digit"],
                "confidence": res["confidence"],
                "bbox": seg["bbox"]
            })
            conf_list.append(res["confidence"])
            
        assembled_str = "".join([d["char"] for d in digit_results if d["char"] is not None])
        
        # Overall confidence: weighted combination of minimum and average
        min_c = min(conf_list) if conf_list else 0.0
        avg_c = sum(conf_list) / max(1, len(conf_list))
        overall_conf = round(0.7 * min_c + 0.3 * avg_c, 4)
        
        val = None
        try:
            val = int(assembled_str)
            success = (overall_conf >= min_confidence) and (len(assembled_str) > 0)
        except ValueError:
            success = False
            
        return {
            "success": success,
            "value": val,
            "number_str": assembled_str,
            "confidence": overall_conf,
            "digits": digit_results,
            "error": None if success else "Low confidence or invalid integer string"
        }


class OperatorClassifier:
    """
    Recognizes arithmetic operators (+, -, ×, ÷) combining structural topology analysis
    and template matching.
    """
    
    OPERATOR_FILES = {
        "+": "add.png",
        "-": "sub.png",
        "×": "mul.png",
        "÷": "div.png"
    }

    def __init__(self, templates_dir: str = "templates/operators"):
        self.templates_dir = templates_dir
        self.templates: Dict[str, np.ndarray] = {}
        self.load_templates()

    def load_templates(self) -> None:
        """Loads normalized operator templates (+, -, ×, ÷)."""
        for sym, fname in self.OPERATOR_FILES.items():
            fpath = os.path.join(self.templates_dir, fname)
            if os.path.exists(fpath):
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                _, bin_tpl = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                self.templates[sym] = bin_tpl
            else:
                self.templates[sym] = np.zeros(TARGET_DIM, dtype=np.uint8)

    @staticmethod
    def analyze_structure(components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes structural topology:
        - Division (÷): 3 components (dot-bar-dot) or 2 components with a central horizontal bar.
        - Minus (-): 1 component with aspect ratio > 1.8.
        - Plus (+) vs Multiply (×): Cross symmetry vs diagonal distribution.
        """
        num_comps = len(components)
        
        # Division Check (÷)
        if num_comps == 3:
            sorted_by_y = sorted(components, key=lambda c: c["bbox"][1])
            mid_comp = sorted_by_y[1]
            if mid_comp["aspect_ratio"] > 1.3:
                return {"symbol": "÷", "bonus": 0.28, "rule": "3-component vertical alignment (dot-bar-dot)"}
        elif num_comps == 2:
            sorted_by_y = sorted(components, key=lambda c: c["bbox"][1])
            for c in sorted_by_y:
                if c["aspect_ratio"] > 1.5:
                    return {"symbol": "÷", "bonus": 0.20, "rule": "2-component division signature"}

        if num_comps >= 1:
            primary = max(components, key=lambda c: c["area"])
            ar = primary["aspect_ratio"]
            
            # Subtraction (-)
            if ar >= 1.85:
                return {"symbol": "-", "bonus": 0.25, "rule": "Single high-aspect horizontal bar (-)"}
                
            # Cross Analysis: + vs ×
            if 0.65 <= ar <= 1.5:
                norm = normalize_character_roi(primary["crop"], TARGET_DIM, margin=3)
                th, tw = norm.shape
                
                # Corner quadrants vs Central axes
                corner_mass = (
                    np.sum(norm[2:10, 2:10] > 0) +
                    np.sum(norm[2:10, tw-10:tw-2] > 0) +
                    np.sum(norm[th-10:th-2, 2:10] > 0) +
                    np.sum(norm[th-10:th-2, tw-10:tw-2] > 0)
                )
                
                axis_mass = (
                    np.sum(norm[2:10, tw//2-3:tw//2+3] > 0) +
                    np.sum(norm[th-10:th-2, tw//2-3:tw//2+3] > 0) +
                    np.sum(norm[th//2-3:th//2+3, 2:10] > 0) +
                    np.sum(norm[th//2-3:th//2+3, tw-10:tw-2] > 0)
                )
                
                if axis_mass > corner_mass * 1.3:
                    return {"symbol": "+", "bonus": 0.22, "rule": "Orthogonal cross symmetry (+)"}
                elif corner_mass > axis_mass * 1.1:
                    return {"symbol": "×", "bonus": 0.22, "rule": "Diagonal stroke symmetry (×)"}

        return {"symbol": None, "bonus": 0.0, "rule": "Generic contour shape"}

    def recognize_operator(self, binary_mask: np.ndarray, min_confidence: float = 0.70) -> Dict[str, Any]:
        """
        Recognizes arithmetic operator from binary mask (+, -, ×, ÷).
        Combines structural analysis and template matching.
        """
        comps = DIPEngine.extract_connected_components(binary_mask, min_pixel_area=15)
        if not comps:
            return {
                "success": False,
                "symbol": None,
                "confidence": 0.0,
                "bbox": None,
                "error": "No operator foreground components found."
            }

        struct_analysis = self.analyze_structure(comps)
        suggested_sym = struct_analysis["symbol"]
        struct_bonus = struct_analysis["bonus"]
        
        # Merge union bbox of all components
        all_x1 = [c["bbox"][0] for c in comps]
        all_y1 = [c["bbox"][1] for c in comps]
        all_x2 = [c["bbox"][0] + c["bbox"][2] for c in comps]
        all_y2 = [c["bbox"][1] + c["bbox"][3] for c in comps]
        
        union_bbox = (min(all_x1), min(all_y1), max(all_x2) - min(all_x1), max(all_y2) - min(all_y1))
        ux, uy, uw, uh = union_bbox
        union_crop = binary_mask[uy:uy+uh, ux:ux+uw]
        norm_roi = normalize_character_roi(union_crop, TARGET_DIM)
        
        # Template Matching
        match_scores = {}
        for sym, tpl in self.templates.items():
            # NCC
            res_map = cv2.matchTemplate(norm_roi, tpl, cv2.TM_CCOEFF_NORMED)
            ncc = max(0.0, float(res_map[0][0]))
            
            # IoU
            inter = np.sum((norm_roi > 0) & (tpl > 0))
            union = np.sum((norm_roi > 0) | (tpl > 0))
            iou = float(inter) / max(1.0, float(union))
            
            base_score = 0.6 * ncc + 0.4 * iou
            
            # Apply structural confirmation bonus
            if sym == suggested_sym:
                base_score = min(1.0, base_score + struct_bonus)
                
            match_scores[sym] = round(base_score, 4)
            
        # Select best candidate
        best_sym = max(match_scores, key=match_scores.get)
        best_conf = match_scores[best_sym]
        
        success = (best_conf >= min_confidence)
        
        return {
            "success": success,
            "symbol": best_sym if success else None,
            "confidence": best_conf,
            "bbox": union_bbox,
            "structural_rule": struct_analysis["rule"],
            "all_scores": match_scores,
            "error": None if success else "Low operator matching confidence."
        }
