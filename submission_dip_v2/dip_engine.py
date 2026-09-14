"""
dip_engine.py - Core Digital Image Processing (DIP) Engine.
Part of DIP-Lab-6 Image-Based Calculator (v2 Architecture).

Implements all required spatial, contrast, morphological, and structural operators:
- Grayscale conversion (Standard Luminance: Y = 0.299R + 0.587G + 0.114B)
- Median Filtering (Edge-preserving impulse & Salt-and-Pepper noise suppression)
- Gaussian Convolution Filtering (Gaussian noise attenuation)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Otsu's Global Optimal Thresholding (Maximizing between-class variance)
- Adaptive Gaussian Local Thresholding (Illumination-gradient robustness)
- Morphological Transformations (Opening for noise cleanup, Closing for stroke bridging)
- Canny Edge Detection with dynamic hysteresis thresholds
- 8-Connectivity Connected Components Analysis (CCA)

Memory Safety:
Strictly treats input image arrays as immutable; all operations return newly allocated copies.
"""

from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np

class DIPEngine:
    """
    Encapsulated Digital Image Processing toolkit providing deterministic,
    immutable transformations for image-based computing.
    """

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Converts input image to an 8-bit single-channel grayscale intensity matrix.
        Uses the standard photometric luminance formula:
            Y = 0.299*R + 0.587*G + 0.114*B
        """
        if image is None:
            raise ValueError("Input image cannot be None.")
        
        # Guard against in-place mutations
        img_copy = image.copy()
        
        if len(img_copy.shape) == 2:
            return img_copy
        elif len(img_copy.shape) == 3:
            channels = img_copy.shape[2]
            if channels == 3:
                return cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            elif channels == 4:
                return cv2.cvtColor(img_copy, cv2.COLOR_BGRA2GRAY)
            elif channels == 1:
                return img_copy[:, :, 0]
        
        raise ValueError(f"Unsupported image dimensions: {img_copy.shape}")

    @staticmethod
    def apply_median(gray_img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Applies a non-linear median filter.
        Highly effective at removing Salt & Pepper noise while preserving sharp character edges.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.medianBlur(gray_img.copy(), kernel_size)

    @staticmethod
    def apply_gaussian(gray_img: np.ndarray, ksize: Tuple[int, int] = (3, 3), sigma: float = 0.8) -> np.ndarray:
        """
        Applies 2D Gaussian convolution smoothing to attenuate high-frequency Gaussian noise.
        """
        kx = ksize[0] if ksize[0] % 2 != 0 else ksize[0] + 1
        ky = ksize[1] if ksize[1] % 2 != 0 else ksize[1] + 1
        return cv2.GaussianBlur(gray_img.copy(), (kx, ky), sigma)

    @staticmethod
    def apply_clahe(gray_img: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Enhances local contrast without over-amplifying background noise.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(gray_img.copy())

    @staticmethod
    def binarize_otsu(gray_img: np.ndarray, invert: bool = True) -> np.ndarray:
        """
        Calculates optimal global threshold T* using Otsu's method (maximizing between-class variance).
        Returns binary mask with foreground pixels = 255 (white) and background = 0 (black).
        """
        thresh_flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray_img.copy(), 0, 255, thresh_flag + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def binarize_adaptive(gray_img: np.ndarray, block_size: int = 15, c_const: int = 4, invert: bool = True) -> np.ndarray:
        """
        Applies adaptive thresholding using a Gaussian-weighted neighborhood.
        Adapts dynamically to non-uniform ambient illumination and shadows.
        """
        thresh_flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        if block_size % 2 == 0:
            block_size += 1
        return cv2.adaptiveThreshold(
            gray_img.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_flag, block_size, c_const
        )

    @staticmethod
    def morphology_open(binary_mask: np.ndarray, ksize: int = 3) -> np.ndarray:
        """
        Morphological Opening = Erosion followed by Dilation.
        Removes isolated foreground noise points and thin parasitic protrusions.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        return cv2.morphologyEx(binary_mask.copy(), cv2.MORPH_OPEN, kernel)

    @staticmethod
    def morphology_close(binary_mask: np.ndarray, ksize: int = 3) -> np.ndarray:
        """
        Morphological Closing = Dilation followed by Erosion.
        Bridges hairline discontinuities, holes, and internal stroke fractures.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        return cv2.morphologyEx(binary_mask.copy(), cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def detect_canny_edges(gray_or_denoised: np.ndarray, low_thresh: Optional[float] = None, high_thresh: Optional[float] = None) -> np.ndarray:
        """
        Extracts structural boundaries using Canny Edge Detection with dynamic Otsu-guided hysteresis thresholds.
        """
        img = gray_or_denoised.copy()
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        if low_thresh is None or high_thresh is None:
            otsu_val, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            high_thresh = float(otsu_val)
            low_thresh = float(otsu_val * 0.5)

        return cv2.Canny(img, int(low_thresh), int(high_thresh))

    @staticmethod
    def extract_connected_components(binary_mask: np.ndarray, min_pixel_area: int = 18) -> List[Dict[str, Any]]:
        """
        Performs 8-connectivity Connected Components Analysis (CCA).
        Extracts bounding boxes, areas, centroids, and binary crops for each valid component.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        components = []
        # Label 0 is background; evaluate labels 1 to num_labels - 1
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_pixel_area:
                continue
                
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[label]
            
            comp_mask = np.uint8(labels == label) * 255
            comp_crop = comp_mask[y:y+h, x:x+w]
            
            aspect_ratio = float(w) / max(1.0, float(h))
            
            components.append({
                "label": label,
                "bbox": (x, y, w, h),
                "area": area,
                "aspect_ratio": aspect_ratio,
                "centroid": (float(cx), float(cy)),
                "mask": comp_mask,
                "crop": comp_crop
            })
            
        return components

    @classmethod
    def filter_stray_edges(cls, canny_edges: np.ndarray, binary_mask: np.ndarray, min_area: int = 20) -> np.ndarray:
        """
        Suppresses spurious edge pixels produced by noise by masking them against
        dilated valid connected components.
        """
        comps = cls.extract_connected_components(binary_mask, min_pixel_area=min_area)
        if not comps:
            return np.zeros_like(canny_edges)
            
        union_mask = np.zeros_like(binary_mask)
        for c in comps:
            union_mask = cv2.bitwise_or(union_mask, c["mask"])
            
        dilated_envelope = cv2.dilate(union_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        return cv2.bitwise_and(canny_edges, dilated_envelope)
