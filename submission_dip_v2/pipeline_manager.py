"""
pipeline_manager.py - Multi-Pipeline DIP Coordinator & Fallback Retries.
Part of DIP-Lab-6 Image-Based Calculator (v2 Architecture).

Implements the multi-pipeline fallback architecture:
- Pipeline Alpha (Impulse Suppression): Median Filter (k=3) -> Otsu -> Morph Close
- Pipeline Beta  (Gaussian Denoising): Gaussian Filter (sigma=0.8) -> Otsu -> Morph Open & Close
- Pipeline Gamma (Contrast Boosting): CLAHE -> Median Filter -> Otsu -> Morph Close

Fallback Policy:
Pipelines are evaluated sequentially. If Pipeline Alpha confidence exceeds the target high threshold
(e.g., 88%), execution terminates early for optimal performance. Otherwise, subsequent pipelines are
evaluated and the candidate yielding the highest confidence is selected.
"""

from typing import Dict, Any, List
import numpy as np
from dip_engine import DIPEngine

class DIPPipelineCoordinator:
    """
    Executes and orchestrates DIP transformation pipelines with automated fallback.
    """

    AVAILABLE_PIPELINES = ["alpha", "beta", "gamma"]

    @classmethod
    def execute_pipeline(cls, image: np.ndarray, pipeline_id: str = "alpha") -> Dict[str, Any]:
        """
        Executes a single DIP pipeline and records every intermediate transformation stage.
        
        Stages recorded:
        1. original
        2. gray
        3. denoised
        4. binary
        5. morph
        6. canny
        7. clean_edges
        """
        gray = DIPEngine.to_grayscale(image)
        
        pid = pipeline_id.lower().strip()
        if pid in ["alpha", "pipeline_1", "1"]:
            # Pipeline Alpha: Median Filter + Otsu + Morph Close
            denoised = DIPEngine.apply_median(gray, kernel_size=3)
            binary = DIPEngine.binarize_otsu(denoised, invert=True)
            morph = DIPEngine.morphology_close(binary, ksize=3)
            pipeline_name = "Pipeline Alpha (Median+Otsu+Close)"

        elif pid in ["beta", "pipeline_2", "2"]:
            # Pipeline Beta: Gaussian Blur + Otsu + Morph Open + Morph Close
            denoised = DIPEngine.apply_gaussian(gray, ksize=(3, 3), sigma=0.8)
            binary = DIPEngine.binarize_otsu(denoised, invert=True)
            opened = DIPEngine.morphology_open(binary, ksize=3)
            morph = DIPEngine.morphology_close(opened, ksize=3)
            pipeline_name = "Pipeline Beta (Gaussian+Otsu+Open+Close)"

        elif pid in ["gamma", "pipeline_3", "3"]:
            # Pipeline Gamma: CLAHE + Median Filter + Otsu + Morph Close
            enhanced = DIPEngine.apply_clahe(gray, clip_limit=2.0)
            denoised = DIPEngine.apply_median(enhanced, kernel_size=3)
            binary = DIPEngine.binarize_otsu(denoised, invert=True)
            morph = DIPEngine.morphology_close(binary, ksize=3)
            pipeline_name = "Pipeline Gamma (CLAHE+Median+Otsu+Close)"

        else:
            # Fallback default
            denoised = DIPEngine.apply_median(gray, kernel_size=3)
            binary = DIPEngine.binarize_otsu(denoised, invert=True)
            morph = binary
            pipeline_name = f"Pipeline Custom ({pid})"

        # Structural Canny Edge Detection & Stray Edge Filtering
        canny = DIPEngine.detect_canny_edges(denoised)
        clean_edges = DIPEngine.filter_stray_edges(canny, morph, min_area=18)

        return {
            "pipeline_id": pid,
            "pipeline_name": pipeline_name,
            "original": image,
            "gray": gray,
            "denoised": denoised,
            "binary": binary,
            "morph": morph,
            "canny": canny,
            "clean_edges": clean_edges
        }
