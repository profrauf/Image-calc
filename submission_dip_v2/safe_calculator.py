"""
safe_calculator.py - High-reliability arithmetic evaluation engine.
Part of DIP-Lab-6 Image-Based Calculator (v2 Architecture).

Guarantees:
1. Zero use of Python's eval() or exec() for security and academic integrity.
2. Explicit handler functions for basic arithmetic operations (+, -, *, /).
3. Graceful handling of division-by-zero with clear mathematical feedback.
4. Clean numerical formatting for integers and floating-point outputs.
"""

from typing import Union, Dict, Any, Callable

Number = Union[int, float]

def op_add(a: float, b: float) -> float:
    """Computes arithmetic addition: a + b."""
    return a + b

def op_subtract(a: float, b: float) -> float:
    """Computes arithmetic subtraction: a - b."""
    return a - b

def op_multiply(a: float, b: float) -> float:
    """Computes arithmetic multiplication: a * b."""
    return a * b

def op_divide(a: float, b: float) -> float:
    """
    Computes arithmetic division: a / b.
    Raises ZeroDivisionError if denominator is zero.
    """
    if abs(b) < 1e-12:
        raise ZeroDivisionError("Division by zero is mathematically undefined.")
    return a / b

class SafeArithmeticEngine:
    """
    Encapsulated arithmetic dispatcher with rigorous input validation
    and clean string formatting.
    """
    
    OPERATIONS: Dict[str, Callable[[float, float], float]] = {
        "+": op_add,
        "add": op_add,
        "plus": op_add,
        "-": op_subtract,
        "sub": op_subtract,
        "minus": op_subtract,
        "−": op_subtract,
        "×": op_multiply,
        "mul": op_multiply,
        "*": op_multiply,
        "x": op_multiply,
        "X": op_multiply,
        "÷": op_divide,
        "div": op_divide,
        "/": op_divide
    }

    CANONICAL_SYMBOLS: Dict[str, str] = {
        "+": "+",
        "add": "+",
        "plus": "+",
        "-": "-",
        "sub": "-",
        "minus": "-",
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

    @staticmethod
    def format_val(value: Number) -> str:
        """Formats a numeric value cleanly: no trailing .0 for whole integers, up to 4 decimals for floats."""
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)):
            if abs(value - round(value)) < 1e-9:
                return str(int(round(value)))
            return f"{value:.4f}".rstrip('0').rstrip('.')
        return str(value)

    @classmethod
    def evaluate(cls, operand1: Number, operator_symbol: str, operand2: Number) -> Dict[str, Any]:
        """
        Safely computes: operand1 [operator_symbol] operand2.
        
        Returns:
            dict containing success, result_value, result_str, formatted_expression, and error.
        """
        clean_op = str(operator_symbol).strip()
        disp_op = cls.CANONICAL_SYMBOLS.get(clean_op, clean_op)
        expr_str = f"{cls.format_val(operand1)} {disp_op} {cls.format_val(operand2)}"

        if clean_op not in cls.OPERATIONS:
            return {
                "success": False,
                "error": f"Unsupported or unknown operator '{operator_symbol}'",
                "expression": expr_str,
                "result_value": None,
                "result_str": "Error",
                "full_equation": f"{expr_str} = Error (Invalid Operator)"
            }

        operation = cls.OPERATIONS[clean_op]

        try:
            val_a = float(operand1)
            val_b = float(operand2)
            calculated_val = operation(val_a, val_b)
            result_str = cls.format_val(calculated_val)

            return {
                "success": True,
                "error": None,
                "expression": expr_str,
                "result_value": calculated_val,
                "result_str": result_str,
                "full_equation": f"{expr_str} = {result_str}"
            }

        except ZeroDivisionError as zde:
            return {
                "success": False,
                "error": str(zde),
                "expression": expr_str,
                "result_value": None,
                "result_str": "Undefined (÷ 0)",
                "full_equation": f"{expr_str} = Undefined (Division by Zero)"
            }
        except Exception as ex:
            return {
                "success": False,
                "error": f"Evaluation error: {str(ex)}",
                "expression": expr_str,
                "result_value": None,
                "result_str": "Calculation Error",
                "full_equation": f"{expr_str} = Error ({str(ex)})"
            }

# Global shorthand function for compatibility
def calculate_expression(num1: Number, op_symbol: str, num2: Number) -> Dict[str, Any]:
    return SafeArithmeticEngine.evaluate(num1, op_symbol, num2)
