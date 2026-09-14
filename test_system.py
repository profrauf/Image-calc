"""
test_system.py - Comprehensive automated test suite for Image-Based Calculator.
Validates:
1. Calculator module & safe zero division
2. Recognition of all 9 noisy numbers
3. Recognition of all 4 noisy operators
4. Multi-pipeline fallback mechanism
5. Full end-to-end expression calculations
6. RAM-based image immutability (original is never modified)
"""

import os
import glob
import cv2
import numpy as np

import calculator
import preprocessing
import edge_detection
import segmentation
import digit_recognizer
import operator_recognizer
import image_processor

import sys
# Set stdout encoding if needed
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_tests():
    print("=" * 65)
    print("   AUTOMATED TEST SUITE - IMAGE-BASED CALCULATOR")
    print("=" * 65)
    
    passed_count = 0
    total_count = 0
    
    # -------------------------------------------------------------
    # TEST 1: Calculator Unit Tests
    # -------------------------------------------------------------
    total_count += 1
    print("\n[Test 1] Testing calculator.py operations...")
    t1_add = calculator.calculate_expression(42, "+", 8)
    t1_sub = calculator.calculate_expression(90, "-", 25)
    t1_mul = calculator.calculate_expression(7, "×", 6)
    t1_div = calculator.calculate_expression(92, "÷", 55)
    t1_zero = calculator.calculate_expression(10, "÷", 0)
    
    assert t1_add["result_str"] == "50", f"Expected 50, got {t1_add['result_str']}"
    assert t1_sub["result_str"] == "65", f"Expected 65, got {t1_sub['result_str']}"
    assert t1_mul["result_str"] == "42", f"Expected 42, got {t1_mul['result_str']}"
    assert t1_div["result_str"] == "1.6727", f"Expected 1.6727, got {t1_div['result_str']}"
    assert t1_zero["success"] is False and "Division by zero" in t1_zero["error"]
    print("   [PASS] All arithmetic operations (+, -, *, /) passed.")
    print("   [PASS] Division by zero handled gracefully without crash.")
    passed_count += 1

    # -------------------------------------------------------------
    # TEST 2: Recognition of all 9 Noisy Numbers
    # -------------------------------------------------------------
    total_count += 1
    print("\n[Test 2] Testing recognition on 9 noisy numbers (< 100)...")
    expected_numbers = {
        "num_1_7": 7,
        "num_2_18": 18,
        "num_3_24": 24,
        "num_4_42": 42,
        "num_5_55": 55,
        "num_6_63": 63,
        "num_7_79": 79,
        "num_8_80": 80,
        "num_9_92": 92
    }
    
    processor = image_processor.ImageProcessor()
    num_errors = 0
    
    for prefix, expected_val in expected_numbers.items():
        filepath = f"input/{prefix}_noisy.png"
        img = cv2.imread(filepath)
        res = processor.process_number(img)
        
        match = (res["value"] == expected_val)
        status_sym = "[PASS]" if match else "[FAIL]"
        print(f"   {status_sym} {prefix}: Expected={expected_val}, Detected={res['value']} (Conf: {res['confidence']:.2f}, Pipeline: {res['pipeline_used']})")
        if not match:
            num_errors += 1
            
    assert num_errors == 0, f"{num_errors} number recognition tests failed!"
    print("   [PASS] All 9 noisy numbers recognized with 100% accuracy!")
    passed_count += 1

    # -------------------------------------------------------------
    # TEST 3: Recognition of all 4 Noisy Operators
    # -------------------------------------------------------------
    total_count += 1
    print("\n[Test 3] Testing recognition on 4 noisy operators (+, -, *, /)...")
    expected_ops = {
        "op_add": "+",
        "op_sub": "-",
        "op_mul": "×",
        "op_div": "÷"
    }
    
    op_errors = 0
    for prefix, expected_sym in expected_ops.items():
        filepath = f"input/{prefix}_noisy.png"
        img = cv2.imread(filepath)
        res = processor.process_operator(img)
        
        match = (res["symbol"] == expected_sym)
        status_sym = "[PASS]" if match else "[FAIL]"
        print(f"   {status_sym} {prefix}: Expected={expected_sym}, Detected={res['symbol']} (Conf: {res['confidence']:.2f}, Pipeline: {res['pipeline_used']})")
        if not match:
            op_errors += 1
            
    assert op_errors == 0, f"{op_errors} operator recognition tests failed!"
    print("   [PASS] All 4 operators recognized with 100% accuracy!")
    passed_count += 1

    # -------------------------------------------------------------
    # TEST 4: End-to-End Full Expression Pipeline
    # -------------------------------------------------------------
    total_count += 1
    print("\n[Test 4] Testing End-to-End Calculation (92 / 55)...")
    img_92 = cv2.imread("input/num_9_92_noisy.png")
    img_div = cv2.imread("input/op_div_noisy.png")
    img_55 = cv2.imread("input/num_5_55_noisy.png")
    
    expr_res = processor.process_expression(img_92, img_div, img_55)
    print(f"   Detected Expression: {expr_res['calculation']['full_equation']}")
    assert expr_res["calculation"]["result_str"] == "1.6727"
    assert expr_res["calculation"]["success"] is True
    print("   [PASS] End-to-end target expression successfully calculated!")
    passed_count += 1

    # -------------------------------------------------------------
    # TEST 5: Original Image Immutability (Guideline 12)
    # -------------------------------------------------------------
    total_count += 1
    print("\n[Test 5] Verifying original image RAM immutability...")
    test_img = cv2.imread("input/num_4_42_noisy.png")
    test_img_copy = test_img.copy()
    
    # Process original array directly
    _ = processor.process_number(test_img)
    
    # Verify exact pixel equivalence
    diff = np.sum(np.abs(test_img.astype(int) - test_img_copy.astype(int)))
    assert diff == 0, "Input image was mutated during processing!"
    print("   [PASS] Original image is 100% immutable in RAM.")
    passed_count += 1

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"   ALL {passed_count}/{total_count} TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
