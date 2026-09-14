import os
import cv2
import numpy as np

# Fixed specifications for standard images and templates
CANVAS_SIZE = (120, 160)  # height, width
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 2.0
FONT_THICKNESS = 4
TEMPLATE_SIZE = (32, 32)  # height, width for normalized template matching

def ensure_dirs():
    dirs = [
        "templates/digits",
        "templates/operators",
        "input",
        "output"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def render_character_image(char_str, canvas_size=CANVAS_SIZE, font_scale=FONT_SCALE, thickness=FONT_THICKNESS):
    """
    Renders a character or number centered on a white canvas (255) with black text (0).
    """
    h, w = canvas_size
    canvas = np.ones((h, w), dtype=np.uint8) * 255
    
    # Custom rendering for division sign if needed
    if char_str == "÷":
        # Draw division: horizontal line with dot above and dot below
        cx, cy = w // 2, h // 2
        line_w = 40
        line_t = thickness
        cv2.line(canvas, (cx - line_w // 2, cy), (cx + line_w // 2, cy), 0, line_t)
        dot_r = max(2, thickness)
        cv2.circle(canvas, (cx, cy - 18), dot_r, 0, -1)
        cv2.circle(canvas, (cx, cy + 18), dot_r, 0, -1)
        return canvas
    
    # Custom rendering for multiplication sign:
    if char_str == "×" or char_str == "x" or char_str == "*":
        cx, cy = w // 2, h // 2
        arm = 20
        cv2.line(canvas, (cx - arm, cy - arm), (cx + arm, cy + arm), 0, thickness)
        cv2.line(canvas, (cx - arm, cy + arm), (cx + arm, cy - arm), 0, thickness)
        return canvas

    if char_str == "-":
        cx, cy = w // 2, h // 2
        arm = 22
        cv2.line(canvas, (cx - arm, cy), (cx + arm, cy), 0, thickness)
        return canvas

    if char_str == "+":
        cx, cy = w // 2, h // 2
        arm = 22
        cv2.line(canvas, (cx - arm, cy), (cx + arm, cy), 0, thickness)
        cv2.line(canvas, (cx, cy - arm), (cx, cy + arm), 0, thickness)
        return canvas

    # For standard text digits:
    (text_w, text_h), baseline = cv2.getTextSize(char_str, FONT, font_scale, thickness)
    x = (w - text_w) // 2
    y = (h + text_h) // 2
    cv2.putText(canvas, char_str, (x, y), FONT, font_scale, 0, thickness, cv2.LINE_AA)
    return canvas

def extract_normalized_template(char_img, target_size=TEMPLATE_SIZE):
    """
    Given a clean grayscale image (white bg, black foreground),
    thresholds to binary (white foreground, black bg),
    finds bounding box of the symbol/digit, crops it, and normalizes into target_size with aspect ratio preserved.
    """
    # Foreground is black (0), background is white (255)
    # Convert to binary where foreground is 255
    binary = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Find bounding box of all non-zero pixels
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros(target_size, dtype=np.uint8)
        
    x, y, w, h = cv2.boundingRect(coords)
    crop = binary[y:y+h, x:x+w]
    
    # Resize preserving aspect ratio into target_size
    th, tw = target_size
    pad_box = np.zeros((th, tw), dtype=np.uint8)
    
    # Compute scale to fit within (th - 4, tw - 4) for a clean margin
    max_dim = max(w, h)
    scale = (min(th, tw) - 4) / max_dim
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Threshold again to keep strictly binary
    resized = cv2.threshold(resized, 100, 255, cv2.THRESH_BINARY)[1]
    
    start_x = (tw - new_w) // 2
    start_y = (th - new_h) // 2
    pad_box[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    return pad_box

def add_gaussian_noise(image, mean=0, sigma=25):
    """Adds controlled Gaussian noise."""
    noisy = image.astype(np.float32)
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
    return noisy

def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """Adds controlled Salt & Pepper noise."""
    noisy = image.copy()
    # Salt (white pixels)
    num_salt = int(np.ceil(salt_prob * image.size))
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[tuple(coords)] = 255
    
    # Pepper (black pixels)
    num_pepper = int(np.ceil(pepper_prob * image.size))
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[tuple(coords)] = 0
    return noisy

def add_blur_noise(image, ksize=(3, 3)):
    """Adds slight blur."""
    return cv2.GaussianBlur(image, ksize, 0)

def generate_all():
    ensure_dirs()
    np.random.seed(42)  # Guideline 17: Fixed Random Seed for reproducibility
    
    print("1. Generating Digits Templates (0..9)...")
    for d in range(10):
        digit_char = str(d)
        clean_img = render_character_image(digit_char)
        template = extract_normalized_template(clean_img)
        cv2.imwrite(f"templates/digits/{d}.png", template)
    
    print("2. Generating Operators Templates (+, -, ×, ÷)...")
    operators = {
        "add": "+",
        "sub": "-",
        "mul": "×",
        "div": "÷"
    }
    for name, op_char in operators.items():
        clean_img = render_character_image(op_char)
        template = extract_normalized_template(clean_img)
        cv2.imwrite(f"templates/operators/{name}.png", template)

    print("3. Generating 9 Test Numbers (< 100) with Controlled Noise...")
    # Generating 9 fixed-seed numbers under 100:
    test_numbers = [7, 18, 24, 42, 55, 63, 79, 80, 92]
    
    for i, num in enumerate(test_numbers, 1):
        num_str = str(num)
        # Using adjusted font scale for 2-digit numbers
        scale = 1.9 if len(num_str) == 1 else 1.6
        clean_num_img = render_character_image(num_str, font_scale=scale)
        cv2.imwrite(f"input/num_{i}_{num}_clean.png", clean_num_img)
        
        # Apply 1 or 2 noise types deterministically
        noisy_img = clean_num_img.copy()
        if i % 3 == 0:
            # Salt & Pepper
            noisy_img = add_salt_and_pepper_noise(noisy_img, salt_prob=0.03, pepper_prob=0.03)
        elif i % 3 == 1:
            # Gaussian Noise
            noisy_img = add_gaussian_noise(noisy_img, mean=0, sigma=25)
        else:
            # Gaussian Noise + slight Salt/Pepper
            noisy_img = add_gaussian_noise(noisy_img, mean=0, sigma=18)
            noisy_img = add_salt_and_pepper_noise(noisy_img, salt_prob=0.015, pepper_prob=0.015)
            
        cv2.imwrite(f"input/num_{i}_{num}_noisy.png", noisy_img)
        print(f"   Generated: num_{i}_{num} (clean & noisy)")

    print("4. Generating Test Operators (+, -, ×, ÷) with Controlled Noise...")
    for name, op_char in operators.items():
        clean_op_img = render_character_image(op_char)
        cv2.imwrite(f"input/op_{name}_clean.png", clean_op_img)
        
        # Apply controlled noise
        noisy_op = add_salt_and_pepper_noise(clean_op_img, salt_prob=0.02, pepper_prob=0.02)
        noisy_op = add_gaussian_noise(noisy_op, mean=0, sigma=15)
        cv2.imwrite(f"input/op_{name}_noisy.png", noisy_op)
        print(f"   Generated: op_{name} (clean & noisy)")

    print("All templates and test datasets generated successfully!")

if __name__ == "__main__":
    generate_all()
