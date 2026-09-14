"""
run_tests.py - Comprehensive Automated Verification Suite.
Part of DIP-Lab-6 Image-Based Calculator (v2 Architecture).

Verifies:
1. Safe calculator operations (+, -, *, /) and zero-division protection without eval().
2. Recognition accuracy across all 9 newly generated noisy test numbers (< 100).
3. Recognition accuracy across all 4 newly generated noisy operators (+, -, *, /).
4. End-to-end composite expression calculation with fallback integration.
5. In-memory array immutability of original images (Guideline 12).
"""

import os
import sys
import cv2
import numpy as np

# Adjust stdout encoding for clean console rendering
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from safe_calculator import SafeArithmeticEngine
from processor import SystemImageProcessor

def run_test_suite():
    print("=" * 70)
    print("   DIP CALCULATOR (v2) - AUTOMATED VERIFICATION SUITE")
    print("=" * 70)
    
    passed_tests = 0
    total_tests = 0
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processor = SystemImageProcessor(base_dir=base_dir)

    # -------------------------------------------------------------
    # Test 1: Safe Arithmetic Engine
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[Test 1] Testing Safe Arithmetic Engine (No eval(), Zero-Division Handling)...")
    r1 = SafeArithmeticEngine.evaluate(46, "+", 37)
    r2 = SafeArithmeticEngine.evaluate(95, "-", 28)
    r3 = SafeArithmeticEngine.evaluate(15, "×", 6)
    r4 = SafeArithmeticEngine.evaluate(95, "÷", 15)
    r5 = SafeArithmeticEngine.evaluate(58, "÷", 0)
    
    assert r1["result_str"] == "83", f"Expected 83, got {r1['result_str']}"
    assert r2["result_str"] == "67", f"Expected 67, got {r2['result_str']}"
    assert r3["result_str"] == "90", f"Expected 90, got {r3['result_str']}"
    assert r4["result_str"] == "6.3333", f"Expected 6.3333, got {r4['result_str']}"
    assert r5["success"] is False and "zero" in r5["error"].lower()
    
    print("   [PASS] Basic arithmetic (+, -, *, /) accurate and formatted.")
    print("   [PASS] Zero-division handled safely without crashes.")
    passed_tests += 1

    # -------------------------------------------------------------
    # Test 2: Recognition of all 9 Noisy Numbers (< 100)
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[Test 2] Testing recognition on 9 newly generated noisy numbers (< 100)...")
    expected_numbers = {
        "num_1_9": 9,
        "num_2_15": 15,
        "num_3_28": 28,
        "num_4_37": 37,
        "num_5_46": 46,
        "num_6_58": 58,
        "num_7_64": 64,
        "num_8_83": 83,
        "num_9_95": 95
    }
    
    num_failures = 0
    for prefix, expected_val in expected_numbers.items():
        img_path = os.path.join(base_dir, "input", f"{prefix}_noisy.png")
        img = cv2.imread(img_path)
        if img is None:
            print(f"   [FAIL] Missing image: {img_path}")
            num_failures += 1
            continue
            
        res = processor.process_number(img)
        match = (res["value"] == expected_val)
        status = "[PASS]" if match else "[FAIL]"
        print(f"   {status} {prefix}: Expected={expected_val}, Detected={res['value']} (Conf: {res['confidence']:.2f}, Pipeline: {res['pipeline_used']})")
        if not match:
            num_failures += 1
            
    assert num_failures == 0, f"{num_failures} number recognition tests failed!"
    print("   [PASS] All 9 noisy numbers recognized with 100% precision.")
    passed_tests += 1

    # -------------------------------------------------------------
    # Test 3: Recognition of all 4 Noisy Operators (+, -, *, /)
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[Test 3] Testing recognition on 4 noisy operators (+, -, *, /)...")
    expected_operators = {
        "op_add": "+",
        "op_sub": "-",
        "op_mul": "×",
        "op_div": "÷"
    }
    
    op_failures = 0
    for prefix, expected_sym in expected_operators.items():
        img_path = os.path.join(base_dir, "input", f"{prefix}_noisy.png")
        img = cv2.imread(img_path)
        if img is None:
            print(f"   [FAIL] Missing image: {img_path}")
            op_failures += 1
            continue
            
        res = processor.process_operator(img)
        match = (res["symbol"] == expected_sym)
        status = "[PASS]" if match else "[FAIL]"
        print(f"   {status} {prefix}: Expected={expected_sym}, Detected={res['symbol']} (Conf: {res['confidence']:.2f}, Pipeline: {res['pipeline_used']})")
        if not match:
            op_failures += 1
            
    assert op_failures == 0, f"{op_failures} operator recognition tests failed!"
    print("   [PASS] All 4 noisy operators recognized with 100% precision.")
    passed_tests += 1

    # -------------------------------------------------------------
    # Test 4: End-to-End Expression Calculation
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[Test 4] Testing End-to-End Expression Calculation (95 / 15)...")
    img_95 = cv2.imread(os.path.join(base_dir, "input", "num_9_95_noisy.png"))
    img_div = cv2.imread(os.path.join(base_dir, "input", "op_div_noisy.png"))
    img_15 = cv2.imread(os.path.join(base_dir, "input", "num_2_15_noisy.png"))
    
    expr_res = processor.process_expression(img_95, img_div, img_15)
    eq_str = expr_res["calculation"]["full_equation"]
    print(f"   Detected Expression: {eq_str}")
    assert expr_res["calculation"]["result_str"] == "6.3333"
    assert expr_res["calculation"]["success"] is True
    print("   [PASS] End-to-end calculation completed successfully.")
    passed_tests += 1

    # -------------------------------------------------------------
    # Test 5: Original Image RAM Immutability
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[Test 5] Verifying original input image RAM immutability...")
    sample_img = cv2.imread(os.path.join(base_dir, "input", "num_5_46_noisy.png"))
    original_copy = sample_img.copy()
    
    # Process through pipeline
    _ = processor.process_number(sample_img)
    
    # Calculate pixel difference
    diff = int(np.sum(np.abs(sample_img.astype(np.int32) - original_copy.astype(np.int32))))
    assert diff == 0, "Input image was mutated in RAM during processing!"
    print("   [PASS] Original image is 100% immutable in RAM (Delta = 0).")
    passed_tests += 1

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"   RESULT: ALL {passed_tests}/{total_tests} TEST SUITES PASSED SUCCESSFULLY (100%)!")
    print("=" * 70)

if __name__ == "__main__":
    run_test_suite()
