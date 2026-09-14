"""
visualizer.py - Visual inspection and pipeline montage tools.
Part of DIP-Lab-6 Image-Based Calculator (v2 Architecture).

Provides:
1. draw_detection_overlays: Bounding box annotation with character labels and confidence percentages.
2. build_pipeline_strip: Horizontal montage displaying the complete sequence of DIP operations:
   [1. Original] -> [2. Grayscale] -> [3. Denoised] -> [4. Binarized] -> [5. Morphology] -> [6. Canny Edges]
"""

from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np

def draw_detection_overlays(image: np.ndarray, result_info: Dict[str, Any]) -> np.ndarray:
    """
    Renders visual bounding boxes and confidence annotations on a color copy of the image.
    """
    if image is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
        
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    
    target_type = result_info.get("type")
    
    if target_type == "number":
        digits = result_info.get("digits", [])
        for d in digits:
            x, y, w, h = d.get("bbox", (0, 0, 0, 0))
            char = d.get("char", "?")
            conf_pct = int(d.get("confidence", 0) * 100)
            
            # Draw green bounding box
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (34, 197, 94), 2)
            
            # Badge background
            label_text = f"{char} ({conf_pct}%)"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            badge_y1 = max(0, y - th - 6)
            badge_y2 = max(th + 6, y)
            cv2.rectangle(canvas, (x, badge_y1), (x + tw + 6, badge_y2), (34, 197, 94), -1)
            cv2.putText(canvas, label_text, (x + 3, badge_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    elif target_type == "operator":
        bbox = result_info.get("bbox")
        if bbox:
            x, y, w, h = bbox
            symbol = result_info.get("symbol", "?")
            conf_pct = int(result_info.get("confidence", 0) * 100)
            
            # Draw blue/cyan bounding box
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (14, 165, 233), 2)
            
            label_text = f"{symbol} ({conf_pct}%)"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            badge_y1 = max(0, y - th - 6)
            badge_y2 = max(th + 6, y)
            cv2.rectangle(canvas, (x, badge_y1), (x + tw + 6, badge_y2), (14, 165, 233), -1)
            cv2.putText(canvas, label_text, (x + 3, badge_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


def build_pipeline_strip(stages: Dict[str, np.ndarray], cell_dim: Tuple[int, int] = (115, 145)) -> np.ndarray:
    """
    Creates a standardized horizontal montage strip showing the 6 DIP transformation steps:
    1. Original Input -> 2. Grayscale -> 3. Denoised Filter -> 4. Binarized Mask -> 5. Morphological Cleanup -> 6. Canny Edge Map
    """
    step_sequence = [
        ("1. Original", stages.get("original")),
        ("2. Grayscale", stages.get("gray")),
        ("3. Denoised", stages.get("denoised")),
        ("4. Threshold", stages.get("binary")),
        ("5. Morphology", stages.get("morph")),
        ("6. Canny Edges", stages.get("canny"))
    ]
    
    target_h, target_w = cell_dim
    banner_height = 24
    panels = []
    
    for title, img_stage in step_sequence:
        if img_stage is None:
            continue
            
        # Convert to 3-channel BGR
        if len(img_stage.shape) == 2:
            cell_bgr = cv2.cvtColor(img_stage, cv2.COLOR_GRAY2BGR)
        else:
            cell_bgr = img_stage.copy()
            
        resized_cell = cv2.resize(cell_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Create dark banner header
        banner = np.zeros((banner_height, target_w, 3), dtype=np.uint8)
        banner[:] = (30, 41, 59)  # Slate-800
        cv2.putText(banner, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (241, 245, 249), 1, cv2.LINE_AA)
        
        # Stack banner over image cell
        panel = np.vstack([banner, resized_cell])
        
        # Border
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (71, 85, 105), 1)
        panels.append(panel)
        
    if not panels:
        return np.zeros((target_h + banner_height, target_w, 3), dtype=np.uint8)
        
    montage = np.hstack(panels)
    return montage
