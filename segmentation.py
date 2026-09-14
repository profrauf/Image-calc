"""
segmentation.py - Segmentation module for extracting individual digits.
Sorts candidate bounding boxes Left -> Right, isolates each digit ROI,
and applies standardized normalization matching the template generator.
"""

import cv2
import numpy as np

DEFAULT_TARGET_SIZE = (32, 32)

def normalize_roi(roi_binary: np.ndarray, target_size=DEFAULT_TARGET_SIZE) -> np.ndarray:
    """
    Standardizes a segmented digit binary mask (255=foreground, 0=background)
    into target_size with aspect-ratio preserved centering.
    Guarantees 100% consistency with generate_templates.py.
    """
    th, tw = target_size
    pad_box = np.zeros((th, tw), dtype=np.uint8)
    
    # Locate exact non-zero boundaries
    coords = cv2.findNonZero(roi_binary)
    if coords is None:
        return pad_box
        
    x, y, w, h = cv2.boundingRect(coords)
    if w <= 0 or h <= 0:
        return pad_box
        
    crop = roi_binary[y:y+h, x:x+w]
    
    # Scale to fit inside (target - 4) with preserved aspect ratio
    max_dim = max(w, h)
    margin = 4
    scale = (min(th, tw) - margin) / float(max_dim)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Ensure sharp binary
    resized = cv2.threshold(resized, 100, 255, cv2.THRESH_BINARY)[1]
    
    start_x = (tw - new_w) // 2
    start_y = (th - new_h) // 2
    pad_box[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    return pad_box

def segment_digits(binary_mask: np.ndarray, min_area=35, min_height=12) -> list[dict]:
    """
    Segments candidate digit regions from a preprocessed binary mask.
    Filters noise, sorts bounding boxes from Left to Right,
    and returns a list of normalized digit items.
    """
    # Use connected components to cleanly separate digits
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    
    candidates = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        
        # Filter out tiny noise specks or border artifacts
        if area < min_area or h < min_height or w < 3:
            continue
            
        comp_mask = np.uint8(labels == label) * 255
        crop = comp_mask[y:y+h, x:x+w]
        
        norm_roi = normalize_roi(crop, DEFAULT_TARGET_SIZE)
        
        candidates.append({
            "label": label,
            "bbox": (x, y, w, h),
            "area": area,
            "aspect_ratio": float(w) / max(1, h),
            "raw_crop": crop,
            "normalized_roi": norm_roi
        })
        
    # Crucial step: Sort from Left to Right by x-coordinate
    candidates.sort(key=lambda item: item["bbox"][0])
    
    return candidates
