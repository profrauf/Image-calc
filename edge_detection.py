"""
edge_detection.py - Edge detection and contour/component extraction module.
Applies Canny edge detection with dynamic thresholds, extracts clean contours,
and provides connected components analysis (vital for multi-element operators like '÷').
"""

import cv2
import numpy as np

def apply_canny(image: np.ndarray, low_threshold=None, high_threshold=None) -> np.ndarray:
    """
    Applies Canny edge detector with automatic thresholding if not provided.
    Accepts grayscale or smoothed image.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    if low_threshold is None or high_threshold is None:
        # Dynamic Otsu-based thresholding
        otsu_val, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_threshold = otsu_val
        low_threshold = otsu_val * 0.5
        
    edges = cv2.Canny(image, int(low_threshold), int(high_threshold))
    return edges

def find_clean_contours(binary_mask: np.ndarray, min_area=25) -> list[dict]:
    """
    Finds external contours and filters out noise contours with area < min_area.
    Returns list of contour info dicts with contour, bounding box (x, y, w, h), and area.
    """
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    clean = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            clean.append({
                "contour": c,
                "bbox": (x, y, w, h),
                "area": area,
                "aspect_ratio": float(w) / max(1, h)
            })
            
    return clean

def find_connected_components(binary_mask: np.ndarray, min_area=20) -> list[dict]:
    """
    Extracts connected components from a binary image (where foreground is 255).
    Crucial for distinguishing operations like '÷' (which has 3 distinct components: dot, bar, dot).
    Returns list of component dicts filtered by min_area.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    
    components = []
    # label 0 is background, so iterate from 1 to num_labels - 1
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[label]
            
            # Mask for this specific component
            comp_mask = np.uint8(labels == label) * 255
            crop = comp_mask[y:y+h, x:x+w]
            
            components.append({
                "label": label,
                "bbox": (x, y, w, h),
                "area": area,
                "centroid": (float(cx), float(cy)),
                "aspect_ratio": float(w) / max(1, h),
                "mask": comp_mask,
                "crop": crop
            })
            
    return components

def clean_edges_with_components(edges: np.ndarray, binary_mask: np.ndarray, min_area=25) -> np.ndarray:
    """
    Cleans edge image by suppressing edges that do not belong to significant connected components.
    Eliminates stray noise edges caused by salt & pepper noise.
    """
    comps = find_connected_components(binary_mask, min_area=min_area)
    valid_mask = np.zeros_like(binary_mask)
    for comp in comps:
        valid_mask = cv2.bitwise_or(valid_mask, comp["mask"])
        
    # Dilate valid mask slightly to cover edge boundaries
    dilated_mask = cv2.dilate(valid_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    clean_edges = cv2.bitwise_and(edges, dilated_mask)
    return clean_edges
