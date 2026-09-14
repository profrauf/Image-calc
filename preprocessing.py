"""
preprocessing.py - Digital Image Processing filters and standardized pipelines.
Includes Grayscale conversion, Gaussian Blur, Median Filter, CLAHE, Otsu & Adaptive Thresholding,
and Morphological operations.
"""

import cv2
import numpy as np

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts image to grayscale if not already."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return image.copy()

def apply_gaussian_blur(gray: np.ndarray, ksize=(3, 3), sigma=0) -> np.ndarray:
    """Applies Gaussian smoothing to suppress Gaussian noise."""
    if ksize[0] % 2 == 0 or ksize[1] % 2 == 0:
        ksize = (ksize[0] | 1, ksize[1] | 1)
    return cv2.GaussianBlur(gray, ksize, sigma)

def apply_median_filter(gray: np.ndarray, ksize=3) -> np.ndarray:
    """Applies Median filtering - highly effective against Salt & Pepper noise."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(gray, ksize)

def apply_clahe(gray: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization for contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)

def apply_threshold(gray: np.ndarray, method="otsu", invert=True) -> np.ndarray:
    """
    Thresholds the image to return a binary image where foreground (digits/operator) is 255 (white)
    and background is 0 (black).
    """
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
        return binary
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, 15, 4
        )
        return binary
    else:  # simple fixed threshold
        _, binary = cv2.threshold(gray, 127, 255, thresh_type)
        return binary

def apply_morphology(binary: np.ndarray, op="open", ksize=3) -> np.ndarray:
    """
    Applies morphological operations:
    - 'open': erosion followed by dilation (removes small noise specks).
    - 'close': dilation followed by erosion (bridges small gaps in strokes).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    if op == "open":
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    elif op == "close":
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary

def run_pipeline(image: np.ndarray, pipeline_name="pipeline_1") -> dict:
    """
    Executes a specific preprocessing pipeline and returns intermediate images at each step
    for visualization and subsequent edge detection/segmentation.
    Output binary has foreground = 255, background = 0.
    """
    gray = to_grayscale(image)
    
    if pipeline_name == "pipeline_1":
        # Pipeline 1: Median filter + Otsu + Morph Close
        # Ideal for Salt & Pepper noise and standard text
        denoised = apply_median_filter(gray, ksize=3)
        binary = apply_threshold(denoised, method="otsu", invert=True)
        morph = apply_morphology(binary, op="close", ksize=3)
        
    elif pipeline_name == "pipeline_2":
        # Pipeline 2: Gaussian Blur + Otsu + Morph Open + Close
        # Ideal for Gaussian noise and slight stroke irregularities
        denoised = apply_gaussian_blur(gray, ksize=(3, 3), sigma=0.8)
        binary = apply_threshold(denoised, method="otsu", invert=True)
        opened = apply_morphology(binary, op="open", ksize=3)
        morph = apply_morphology(opened, op="close", ksize=3)
        
    elif pipeline_name == "pipeline_3":
        # Pipeline 3: Median filter + CLAHE + Otsu + Morph Close
        # Ideal for combined noise or lower contrast images
        denoised = apply_median_filter(gray, ksize=3)
        enhanced = apply_clahe(denoised, clip_limit=2.0)
        binary = apply_threshold(enhanced, method="otsu", invert=True)
        morph = apply_morphology(binary, op="close", ksize=3)
        
    else:
        # Default fallback
        denoised = apply_median_filter(gray, ksize=3)
        binary = apply_threshold(denoised, method="otsu", invert=True)
        morph = binary

    return {
        "pipeline_name": pipeline_name,
        "original": image,
        "gray": gray,
        "denoised": denoised,
        "binary": binary,
        "morph": morph
    }
