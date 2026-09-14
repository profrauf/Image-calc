"""
image_processor.py - Unified orchestrator for DIP pipelines, fallback retries,
and full expression processing.
Executes Pipeline 1 -> Pipeline 2 -> Pipeline 3 fallback strategy
to pick the highest confidence result.
"""

import cv2
import numpy as np
from preprocessing import run_pipeline
from edge_detection import apply_canny, clean_edges_with_components
from digit_recognizer import DigitRecognizer, DEFAULT_CONFIDENCE_THRESHOLD
from operator_recognizer import OperatorRecognizer
from calculator import calculate_expression

class ImageProcessor:
    def __init__(self, templates_digits_dir="templates/digits", templates_ops_dir="templates/operators"):
        self.digit_recognizer = DigitRecognizer(templates_digits_dir)
        self.operator_recognizer = OperatorRecognizer(templates_ops_dir)
        self.pipeline_names = ["pipeline_1", "pipeline_2", "pipeline_3"]
        self.target_conf_threshold = DEFAULT_CONFIDENCE_THRESHOLD

    def process_number(self, image: np.ndarray, min_conf=None) -> dict:
        """
        Processes a number image through the DIP pipeline with fallback retries.
        Tries Pipeline 1 -> 2 -> 3 if confidence is below threshold,
        and selects the candidate with the highest confidence.
        """
        threshold = min_conf if min_conf is not None else self.target_conf_threshold
        best_result = None
        highest_conf = -1.0
        
        for p_name in self.pipeline_names:
            prep_data = run_pipeline(image, pipeline_name=p_name)
            edges = apply_canny(prep_data["denoised"])
            clean_edges = clean_edges_with_components(edges, prep_data["morph"])
            
            rec_res = self.digit_recognizer.recognize_image(prep_data["morph"], min_conf=threshold)
            
            candidate = {
                "type": "number",
                "pipeline_used": p_name,
                "success": rec_res["success"],
                "value": rec_res["value"],
                "string": rec_res["string"],
                "confidence": rec_res["confidence"],
                "digits": rec_res["digits"],
                "stages": {
                    "original": prep_data["original"],
                    "gray": prep_data["gray"],
                    "denoised": prep_data["denoised"],
                    "binary": prep_data["binary"],
                    "morph": prep_data["morph"],
                    "canny": edges,
                    "clean_edges": clean_edges
                }
            }
            
            if rec_res["confidence"] > highest_conf:
                highest_conf = rec_res["confidence"]
                best_result = candidate
                
            # If we achieved very high confidence, no need to execute remaining pipelines
            if rec_res["success"] and rec_res["confidence"] >= 0.88:
                break
                
        return best_result

    def process_operator(self, image: np.ndarray, min_conf=None) -> dict:
        """
        Processes an operator image through the DIP pipeline with fallback retries.
        Selects the result with the highest confidence across pipelines.
        """
        threshold = min_conf if min_conf is not None else self.target_conf_threshold
        best_result = None
        highest_conf = -1.0
        
        for p_name in self.pipeline_names:
            prep_data = run_pipeline(image, pipeline_name=p_name)
            edges = apply_canny(prep_data["denoised"])
            clean_edges = clean_edges_with_components(edges, prep_data["morph"])
            
            rec_res = self.operator_recognizer.recognize_image(prep_data["morph"], min_conf=threshold)
            
            candidate = {
                "type": "operator",
                "pipeline_used": p_name,
                "success": rec_res["success"],
                "symbol": rec_res["symbol"],
                "confidence": rec_res["confidence"],
                "bbox": rec_res.get("bbox"),
                "geometry": rec_res.get("geometry"),
                "all_scores": rec_res.get("all_scores"),
                "stages": {
                    "original": prep_data["original"],
                    "gray": prep_data["gray"],
                    "denoised": prep_data["denoised"],
                    "binary": prep_data["binary"],
                    "morph": prep_data["morph"],
                    "canny": edges,
                    "clean_edges": clean_edges
                }
            }
            
            if rec_res["confidence"] > highest_conf:
                highest_conf = rec_res["confidence"]
                best_result = candidate
                
            if rec_res["success"] and rec_res["confidence"] >= 0.90:
                break
                
        return best_result

    def process_expression(self, img_num1: np.ndarray, img_op: np.ndarray, img_num2: np.ndarray) -> dict:
        """
        Orchestrates full 3-image calculation:
        1. Recognizes number 1
        2. Recognizes operator
        3. Recognizes number 2
        4. Calculates mathematical expression without eval()
        """
        res_num1 = self.process_number(img_num1)
        res_op = self.process_operator(img_op)
        res_num2 = self.process_number(img_num2)
        
        # Check if all 3 parts were recognized
        all_recognized = (
            res_num1["value"] is not None and
            res_op["symbol"] is not None and
            res_num2["value"] is not None
        )
        
        calc_res = None
        if all_recognized:
            calc_res = calculate_expression(
                res_num1["value"],
                res_op["symbol"],
                res_num2["value"]
            )
        else:
            missing = []
            if res_num1["value"] is None: missing.append("First Number")
            if res_op["symbol"] is None: missing.append("Operator")
            if res_num2["value"] is None: missing.append("Second Number")
            
            calc_res = {
                "success": False,
                "error": f"Recognition incomplete for: {', '.join(missing)}",
                "expression": f"{res_num1['string'] or '?'} {res_op['symbol'] or '?'} {res_num2['string'] or '?'}",
                "result_str": "N/A",
                "result_val": None,
                "full_equation": "Incomplete Expression"
            }
            
        overall_confidence = round(
            float(res_num1["confidence"] * 0.35 + res_op["confidence"] * 0.30 + res_num2["confidence"] * 0.35),
            4
        )
        
        return {
            "num1": res_num1,
            "op": res_op,
            "num2": res_num2,
            "calculation": calc_res,
            "overall_confidence": overall_confidence
        }
