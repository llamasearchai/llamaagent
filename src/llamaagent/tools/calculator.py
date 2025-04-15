"""
Calculator tool implementation
"""

import logging
import math
import operator
import re
from typing import Any, Dict, List, Optional

from llamaagent.core.tool import Tool

logger = logging.getLogger(__name__)


class CalculatorTool(Tool):
    """
    Tool for performing mathematical calculations
    """

    name: str = "calculator"
    description: str = "Perform mathematical calculations and evaluate expressions"
    keywords: List[str] = ["calculate", "math", "compute", "solve", "equation"]

    # Dictionary of available operators
    OPERATORS = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "^": operator.pow,
        "**": operator.pow,
        "%": operator.mod,
    }

    # Dictionary of available functions
    FUNCTIONS = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "abs": abs,
        "log": math.log10,
        "ln": math.log,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "round": round,
    }

    # Constants
    CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
    }

    def run(self, input: str, **kwargs) -> str:
        """
        Evaluate a mathematical expression

        Args:
            input: The mathematical expression to evaluate
            **kwargs: Additional parameters

        Returns:
            str: The result of the calculation
        """
        logger.info(f"Calculating: {input}")

        try:
            # Clean the input
            expression = self._clean_expression(input)

            # Check if it's a simple arithmetic expression
            if self._is_simple_arithmetic(expression):
                result = self._evaluate_arithmetic(expression)
                return f"Result: {result}"

            # For more complex expressions, use the safer eval approach
            result = self._safe_eval(expression)
            return f"Result: {result}"

        except Exception as e:
            logger.error(f"Error calculating: {e}")
            return f"Error: {str(e)}"

    def _clean_expression(self, expression: str) -> str:
        """
        Clean the input expression
        """
        # Extract the actual mathematical expression if embedded in text
        math_pattern = r"(-?\d+\.?\d*[-+*/^()%\s]*)+|(\b(sin|cos|tan|sqrt|abs|log|ln|exp|floor|ceil|round)\b\s*\([^)]*\))"
        matches = re.findall(math_pattern, expression)

        if matches:
            # Join all matched parts
            cleaned = "".join(m[0] for m in matches if m[0])

            # If no matches with operations, look for functions
            if not cleaned:
                cleaned = "".join(m[2] for m in matches if m[2])

            # If still no matches, try to extract numbers
            if not cleaned:
                numbers = re.findall(r"-?\d+\.?\d*", expression)
                if numbers:
                    cleaned = numbers[0]
        else:
            # If no clear match, just use the input as is
            cleaned = expression

        # Replace text versions of operators with symbols
        word_to_op = {
            "plus": "+",
            "minus": "-",
            "times": "*",
            "multiplied by": "*",
            "divided by": "/",
            "power": "^",
            "mod": "%",
            "modulo": "%",
        }

        for word, op in word_to_op.items():
            cleaned = re.sub(r"\b" + word + r"\b", op, cleaned, flags=re.IGNORECASE)

        # Replace '^' with '**' for exponentiation
        cleaned = cleaned.replace("^", "**")

        return cleaned

    def _is_simple_arithmetic(self, expression: str) -> bool:
        """
        Check if the expression is a simple arithmetic expression
        """
        # Simple arithmetic expressions contain only numbers and basic operators
        return bool(re.match(r"^[\d\s\+\-\*\/\(\)\.]+$", expression))

    def _evaluate_arithmetic(self, expression: str) -> float:
        """
        Evaluate a simple arithmetic expression
        """
        # Replace ** with ^ for parsing
        expression = expression.replace("**", "^")

        # This is a very simple and unsafe implementation
        # In a real application, you would use a proper parsing algorithm
        # or a library like sympy

        # For the sake of this example, we'll use eval(), but
        # this is generally not recommended for security reasons
        # Replace ^ back to ** for Python evaluation
        expression = expression.replace("^", "**")
        return eval(expression)

    def _safe_eval(self, expression: str) -> float:
        """
        Safely evaluate a mathematical expression

        This is safer than using raw eval() but still has limitations.
        For a production system, use a proper math library like sympy.
        """
        # Replace function names with safe references
        for func_name in self.FUNCTIONS:
            expression = re.sub(
                r"\b" + func_name + r"\b\s*\(",
                f"self.FUNCTIONS['{func_name}'](",
                expression,
            )

        # Replace constants with their values
        for const_name, const_value in self.CONSTANTS.items():
            expression = re.sub(
                r"\b" + const_name + r"\b", str(const_value), expression
            )

        # Define a restricted local environment for evaluation
        local_env = {"self": self, "math": math}

        # Add supported mathematical functions to the environment
        for func_name, func in self.FUNCTIONS.items():
            local_env[func_name] = func

        # Add constants to the environment
        for const_name, const_value in self.CONSTANTS.items():
            local_env[const_name] = const_value

        # Evaluate the expression in the restricted environment
        return eval(expression, {"__builtins__": {}}, local_env)
