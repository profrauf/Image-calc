"""
processor.py - Unified Orchestration Engine for DIP Processing & Recognition.
Part of DIP-Lab-6 Image-Based Calculator (v2 Architecture).

Provides high-level interfaces for:
1. Multi-pipeline fallback processing for numbers.
2. Multi-pipeline fallback processing for arithmetic operators.
3. Complete 3-image mathematical expression processing:
   [Number 1] [Operator] [Number 2] -> Safe Arithmetic Calculation
"""

from typing import Dict, Any, Optional
import os
import cv2
import numpy as np

from pipeline_manager import DIPPipelineCoordinator
from character_recognizer import DigitSegmenterAndRecognizer, OperatorClassifier
from safe_calculator import SafeArithmeticEngine

class SystemImageProcessor:
    """
    Unified coordinator connecting DIP pipelines, recognition engines,
    and safe arithmetic evaluation.
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
        digits_dir = os.path.join(base_dir, "templates", "digits")
        ops_dir = os.path.join(base_dir, "templates", "operators")
        
        self.digit_engine = DigitSegmenterAndRecognizer(templates_dir=digits_dir)
        self.operator_engine = OperatorClassifier(templates_dir=ops_dir)
        self.pipelines = ["alpha", "beta", "gamma"]

    def process_number(self, image: np.ndarray, target_min_conf: float = 0.70) -> Dict[str, Any]:
        """
        Processes a number image through multi-pipeline fallback (Alpha -> Beta -> Gamma).
        Selects the candidate producing the highest recognition confidence.
        """
        best_candidate = None
        highest_conf = -1.0
        
        for p_id in self.pipelines:
            prep = DIPPipelineCoordinator.execute_pipeline(image, pipeline_id=p_id)
            rec = self.digit_engine.recognize_number(prep["morph"], min_confidence=target_min_conf)
            
            candidate = {
                "type": "number",
                "pipeline_used": prep["pipeline_name"],
                "pipeline_id": p_id,
                "success": rec["success"],
                "value": rec["value"],
                "string": rec["number_str"],
                "confidence": rec["confidence"],
                "digits": rec["digits"],
                "stages": prep
            }
            
            if rec["confidence"] > highest_conf:
                highest_conf = rec["confidence"]
                best_candidate = candidate
                
            # If high confidence is achieved, terminate early
            if rec["success"] and rec["confidence"] >= 0.88:
                break
                
        return best_candidate

    def process_operator(self, image: np.ndarray, target_min_conf: float = 0.70) -> Dict[str, Any]:
        """
        Processes an operator image through multi-pipeline fallback (Alpha -> Beta -> Gamma).
        Selects the candidate producing the highest recognition confidence.
        """
        best_candidate = None
        highest_conf = -1.0
        
        for p_id in self.pipelines:
            prep = DIPPipelineCoordinator.execute_pipeline(image, pipeline_id=p_id)
            rec = self.operator_engine.recognize_operator(prep["morph"], min_confidence=target_min_conf)
            
            candidate = {
                "type": "operator",
                "pipeline_used": prep["pipeline_name"],
                "pipeline_id": p_id,
                "success": rec["success"],
                "symbol": rec["symbol"],
                "confidence": rec["confidence"],
                "bbox": rec.get("bbox"),
                "structural_rule": rec.get("structural_rule"),
                "all_scores": rec.get("all_scores"),
                "stages": prep
            }
            
            if rec["confidence"] > highest_conf:
                highest_conf = rec["confidence"]
                best_candidate = candidate
                
            if rec["success"] and rec["confidence"] >= 0.90:
                break
                
        return best_candidate

    def process_expression(self, img_num1: np.ndarray, img_op: np.ndarray, img_num2: np.ndarray) -> Dict[str, Any]:
        """
        Processes full 3-image mathematical calculation:
        1. Recognizes Number 1
        2. Recognizes Operator (+, -, ×, ÷)
        3. Recognizes Number 2
        4. Calculates mathematical expression without eval()
        """
        res_num1 = self.process_number(img_num1)
        res_op = self.process_operator(img_op)
        res_num2 = self.process_number(img_num2)
        
        is_complete = (
            res_num1["value"] is not None and
            res_op["symbol"] is not None and
            res_num2["value"] is not None
        )
        
        if is_complete:
            math_res = SafeArithmeticEngine.evaluate(
                res_num1["value"],
                res_op["symbol"],
                res_num2["value"]
            )
        else:
            missing_items = []
            if res_num1["value"] is None: missing_items.append("First Operand")
            if res_op["symbol"] is None: missing_items.append("Operator")
            if res_num2["value"] is None: missing_items.append("Second Operand")
            
            math_res = {
                "success": False,
                "error": f"Incomplete detection: {', '.join(missing_items)}",
                "expression": f"{res_num1['string'] or '?'} {res_op['symbol'] or '?'} {res_num2['string'] or '?'}",
                "result_str": "N/A",
                "result_value": None,
                "full_equation": "Incomplete Expression"
            }
            
        combined_confidence = round(
            float(res_num1["confidence"] * 0.35 + res_op["confidence"] * 0.30 + res_num2["confidence"] * 0.35),
            4
        )
        
        return {
            "num1": res_num1,
            "op": res_op,
            "num2": res_num2,
            "calculation": math_res,
            "overall_confidence": combined_confidence
        }
