"""
dataset_generator.py - Automated Template & Test Dataset Generation.
Part of DIP-Lab-6 Image-Based Calculator (v2 Architecture).

Generates:
1. Normalized templates for digits (0..9) and operators (+, -, ×, ÷).
2. 9 Brand-New Test Numbers (< 100): [9, 15, 28, 37, 46, 58, 64, 83, 95].
3. Clean and Noisy variants with controlled Gaussian, Salt & Pepper, and Mixed noise.
4. Reproducible deterministic generation via fixed random seed (Guideline 17).
"""

import os
import cv2
import numpy as np

CANVAS_SHAPE = (120, 160)  # Height, Width
TARGET_TEMPLATE_SIZE = (36, 36)
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX

def ensure_storage_directories(base_dir="."):
    """Ensures necessary directory tree exists."""
    subdirs = [
        os.path.join(base_dir, "templates", "digits"),
        os.path.join(base_dir, "templates", "operators"),
        os.path.join(base_dir, "input"),
        os.path.join(base_dir, "output")
    ]
    for d in subdirs:
        os.makedirs(d, exist_ok=True)

def render_symbol_canvas(symbol_text: str, canvas_shape=CANVAS_SHAPE, font_scale=2.0, thickness=4) -> np.ndarray:
    """
    Renders black glyphs on a pure white background (255).
    Features specialized structural rendering for division and operators.
    """
    h, w = canvas_shape
    canvas = np.ones((h, w), dtype=np.uint8) * 255
    cx, cy = w // 2, h // 2
    
    # Custom Division Sign (÷)
    if symbol_text == "÷":
        bar_half_len = 22
        cv2.line(canvas, (cx - bar_half_len, cy), (cx + bar_half_len, cy), 0, thickness)
        dot_radius = max(3, thickness)
        cv2.circle(canvas, (cx, cy - 18), dot_radius, 0, -1)
        cv2.circle(canvas, (cx, cy + 18), dot_radius, 0, -1)
        return canvas

    # Custom Multiplication Sign (×)
    if symbol_text in ["×", "x", "*"]:
        arm = 20
        cv2.line(canvas, (cx - arm, cy - arm), (cx + arm, cy + arm), 0, thickness, cv2.LINE_AA)
        cv2.line(canvas, (cx - arm, cy + arm), (cx + arm, cy - arm), 0, thickness, cv2.LINE_AA)
        return canvas

    # Custom Subtraction Sign (-)
    if symbol_text in ["-", "−"]:
        bar_len = 24
        cv2.line(canvas, (cx - bar_len, cy), (cx + bar_len, cy), 0, thickness)
        return canvas

    # Custom Addition Sign (+)
    if symbol_text == "+":
        arm = 22
        cv2.line(canvas, (cx - arm, cy), (cx + arm, cy), 0, thickness)
        cv2.line(canvas, (cx, cy - arm), (cx, cy + arm), 0, thickness)
        return canvas

    # Standard alphanumeric digits
    if len(symbol_text) > 1 and symbol_text.isdigit():
        spacing = 10
        sizes = [cv2.getTextSize(ch, DEFAULT_FONT, font_scale, thickness)[0] for ch in symbol_text]
        total_w = sum(s[0] for s in sizes) + spacing * (len(symbol_text) - 1)
        max_h = max(s[1] for s in sizes)
        curr_x = (w - total_w) // 2
        ty = (h + max_h) // 2
        for ch, (dw, dh) in zip(symbol_text, sizes):
            cv2.putText(canvas, ch, (curr_x, ty), DEFAULT_FONT, font_scale, 0, thickness, cv2.LINE_AA)
            curr_x += dw + spacing
        return canvas

    (tw, th), _ = cv2.getTextSize(symbol_text, DEFAULT_FONT, font_scale, thickness)
    tx = (w - tw) // 2
    ty = (h + th) // 2
    cv2.putText(canvas, symbol_text, (tx, ty), DEFAULT_FONT, font_scale, 0, thickness, cv2.LINE_AA)
    return canvas

def extract_template(canvas: np.ndarray, target_dim=TARGET_TEMPLATE_SIZE, margin=4) -> np.ndarray:
    """
    Extracts foreground glyph from white-canvas, centers it in target_dim binary mask.
    """
    binary = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY_INV)[1]
    coords = cv2.findNonZero(binary)
    th, tw = target_dim
    out_box = np.zeros((th, tw), dtype=np.uint8)
    
    if coords is None:
        return out_box
        
    x, y, w, h = cv2.boundingRect(coords)
    crop = binary[y:y+h, x:x+w]
    max_dim = max(w, h)
    scale = float(min(th, tw) - (margin * 2)) / float(max_dim)
    
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    _, resized_bin = cv2.threshold(resized, 100, 255, cv2.THRESH_BINARY)
    
    sx = (tw - nw) // 2
    sy = (th - nh) // 2
    out_box[sy:sy+nh, sx:sx+nw] = resized_bin
    return out_box

def inject_gaussian_noise(image: np.ndarray, mean=0, sigma=24) -> np.ndarray:
    """Injects controlled Gaussian noise."""
    noisy = image.astype(np.float32)
    noise = np.random.normal(mean, sigma, image.shape)
    return np.clip(noisy + noise, 0, 255).astype(np.uint8)

def inject_salt_and_pepper_noise(image: np.ndarray, salt_ratio=0.025, pepper_ratio=0.025) -> np.ndarray:
    """Injects controlled Salt & Pepper impulse noise."""
    noisy = image.copy()
    # Salt (white pixels = 255)
    num_salt = int(np.ceil(salt_ratio * image.size))
    coords_s = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[tuple(coords_s)] = 255
    
    # Pepper (black pixels = 0)
    num_pepper = int(np.ceil(pepper_ratio * image.size))
    coords_p = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[tuple(coords_p)] = 0
    return noisy

def generate_full_dataset(base_dir="."):
    """Generates all templates and test datasets."""
    ensure_storage_directories(base_dir)
    np.random.seed(101)  # Fixed reproducible seed
    
    print("[1/4] Generating Digits Templates (0..9)...")
    for d in range(10):
        d_char = str(d)
        clean = render_symbol_canvas(d_char)
        tpl = extract_template(clean)
        cv2.imwrite(os.path.join(base_dir, "templates", "digits", f"{d}.png"), tpl)
        
    print("[2/4] Generating Operators Templates (+, -, ×, ÷)...")
    operators = {
        "add": "+",
        "sub": "-",
        "mul": "×",
        "div": "÷"
    }
    for op_name, op_sym in operators.items():
        clean = render_symbol_canvas(op_sym)
        tpl = extract_template(clean)
        cv2.imwrite(os.path.join(base_dir, "templates", "operators", f"{op_name}.png"), tpl)

    print("[3/4] Generating 9 New Test Numbers (< 100) with Controlled Noise...")
    # Fresh set of 9 numbers under 100:
    test_numbers = [9, 15, 28, 37, 46, 58, 64, 83, 95]
    
    for idx, num in enumerate(test_numbers, 1):
        num_str = str(num)
        fscale = 2.0 if len(num_str) == 1 else 1.6
        clean_img = render_symbol_canvas(num_str, font_scale=fscale)
        clean_path = os.path.join(base_dir, "input", f"num_{idx}_{num}_clean.png")
        cv2.imwrite(clean_path, clean_img)
        
        # Noise injection pattern
        noisy_img = clean_img.copy()
        if idx % 3 == 1:
            # Gaussian Noise
            noisy_img = inject_gaussian_noise(noisy_img, mean=0, sigma=25)
        elif idx % 3 == 2:
            # Salt & Pepper Noise
            noisy_img = inject_salt_and_pepper_noise(noisy_img, salt_ratio=0.03, pepper_ratio=0.03)
        else:
            # Mixed Noise: Gaussian + subtle Salt&Pepper
            noisy_img = inject_gaussian_noise(noisy_img, mean=0, sigma=18)
            noisy_img = inject_salt_and_pepper_noise(noisy_img, salt_ratio=0.015, pepper_ratio=0.015)
            
        noisy_path = os.path.join(base_dir, "input", f"num_{idx}_{num}_noisy.png")
        cv2.imwrite(noisy_path, noisy_img)
        print(f"   Generated: num_{idx}_{num} (clean & noisy)")

    print("[4/4] Generating Test Operators (+, -, ×, ÷) with Controlled Noise...")
    for op_name, op_sym in operators.items():
        clean_img = render_symbol_canvas(op_sym)
        cv2.imwrite(os.path.join(base_dir, "input", f"op_{op_name}_clean.png"), clean_img)
        
        # Noise
        noisy_img = inject_salt_and_pepper_noise(clean_img, salt_ratio=0.02, pepper_ratio=0.02)
        noisy_img = inject_gaussian_noise(noisy_img, mean=0, sigma=16)
        cv2.imwrite(os.path.join(base_dir, "input", f"op_{op_name}_noisy.png"), noisy_img)
        print(f"   Generated: op_{op_name} (clean & noisy)")

    print("\nAll templates and datasets generated successfully in:", base_dir)

if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    generate_full_dataset(target_dir)
