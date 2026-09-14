"""
visualization.py - Visual inspection and pipeline display utilities.
Draws bounding boxes, labels, and creates horizontal step-by-step montages
of the Digital Image Processing pipeline (Original -> Gray -> Denoised -> Threshold -> Morph -> Canny).
"""

import cv2
import numpy as np

def draw_annotations(image: np.ndarray, result_data: dict) -> np.ndarray:
    """Draws detected bounding boxes, characters, and confidence on a copy of the image."""
    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    
    if result_data.get("type") == "number":
        digits = result_data.get("digits", [])
        for d in digits:
            x, y, w, h = d["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 180, 0), 2)
            label = f"{d['char']} ({int(d['confidence']*100)}%)"
            cv2.putText(annotated, label, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 0), 1)
            
    elif result_data.get("type") == "operator":
        bbox = result_data.get("bbox")
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (180, 0, 0), 2)
            label = f"{result_data.get('symbol')} ({int(result_data.get('confidence', 0)*100)}%)"
            cv2.putText(annotated, label, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 0, 0), 1)
            
    return annotated

def create_pipeline_montage(stages: dict, cell_size=(110, 140)) -> np.ndarray:
    """
    Creates a step-by-step horizontal strip showing:
    [1. Original] -> [2. Grayscale] -> [3. Denoised] -> [4. Binary] -> [5. Morph] -> [6. Canny Edges]
    """
    steps = [
        ("1. Original", stages.get("original")),
        ("2. Grayscale", stages.get("gray")),
        ("3. Denoised", stages.get("denoised")),
        ("4. Threshold", stages.get("binary")),
        ("5. Morphology", stages.get("morph")),
        ("6. Canny", stages.get("canny"))
    ]
    
    target_h, target_w = cell_size
    cells = []
    
    for title, img in steps:
        if img is None:
            continue
        # Ensure 3 channels
        if len(img.shape) == 2:
            cell = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            cell = img.copy()
            
        resized = cv2.resize(cell, (target_w, target_h))
        
        # Add title header banner
        banner = np.zeros((24, target_w, 3), dtype=np.uint8) + 40
        cv2.putText(banner, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Combine banner + image cell
        combined_cell = np.vstack([banner, resized])
        # Border
        cv2.rectangle(combined_cell, (0, 0), (combined_cell.shape[1]-1, combined_cell.shape[0]-1), (100, 100, 100), 1)
        cells.append(combined_cell)
        
    if not cells:
        return np.zeros((100, 100, 3), dtype=np.uint8)
        
    montage = np.hstack(cells)
    return montage
