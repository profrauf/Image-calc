"""
calculator.py - Pure mathematical evaluation module.
Safe evaluation of arithmetic operations without using eval().
Handles division by zero gracefully and formats output cleanly.
"""

def add(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b

def multiply(a: float, b: float) -> float:
    return a * b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("Error: Division by zero")
    return a / b

OPERATORS = {
    "+": add,
    "add": add,
    "-": subtract,
    "sub": subtract,
    "−": subtract,
    "×": multiply,
    "mul": multiply,
    "*": multiply,
    "x": multiply,
    "X": multiply,
    "÷": divide,
    "div": divide,
    "/": divide
}

DISPLAY_SYMBOLS = {
    "+": "+",
    "add": "+",
    "-": "-",
    "sub": "-",
    "−": "-",
    "×": "×",
    "mul": "×",
    "*": "×",
    "x": "×",
    "X": "×",
    "÷": "÷",
    "div": "÷",
    "/": "÷"
}

def format_number(val: float) -> str:
    """Formats a number cleanly: no trailing .0 for integers, up to 4 decimal places for floats."""
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    formatted = f"{val:.4f}".rstrip('0').rstrip('.')
    return formatted

def calculate_expression(num1: int | float, op_symbol: str, num2: int | float):
    """
    Safely calculates num1 [op] num2 without using eval().
    Returns a dict with status, result_str, formatted_expr, error_msg.
    """
    op_key = str(op_symbol).strip()
    if op_key not in OPERATORS:
        return {
            "success": False,
            "error": f"Invalid or unrecognized operator: '{op_symbol}'",
            "expression": f"{num1} {op_symbol} {num2}",
            "result_str": None,
            "result_val": None
        }

    func = OPERATORS[op_key]
    disp_op = DISPLAY_SYMBOLS.get(op_key, op_key)
    expr_str = f"{format_number(num1)} {disp_op} {format_number(num2)}"

    try:
        res = func(float(num1), float(num2))
        res_str = format_number(res)
        return {
            "success": True,
            "error": None,
            "expression": expr_str,
            "result_str": res_str,
            "result_val": res,
            "full_equation": f"{expr_str} = {res_str}"
        }
    except ZeroDivisionError as zde:
        return {
            "success": False,
            "error": str(zde),
            "expression": expr_str,
            "result_str": "Undefined (ZeroDivision)",
            "result_val": None,
            "full_equation": f"{expr_str} = Undefined (Division by zero)"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "expression": expr_str,
            "result_str": "Error",
            "result_val": None,
            "full_equation": f"{expr_str} = Error"
        }
