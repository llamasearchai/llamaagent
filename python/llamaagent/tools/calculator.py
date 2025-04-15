"""
Calculator tool for performing mathematical operations.
"""

import ast
import logging
import math
import operator
from typing import Any, Dict, Optional, Union

from .base import BaseTool

logger = logging.getLogger(__name__)


class Calculator(BaseTool):
    """
    Tool for evaluating mathematical expressions safely.

    This tool can handle basic arithmetic, mathematical functions,
    and complex expressions. It uses a restricted environment to
    prevent execution of arbitrary code.
    """

    # Allowed operators
    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Allowed functions and constants
    _ALLOWED_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "pi": math.pi,
        "e": math.e,
        "degrees": math.degrees,
        "radians": math.radians,
    }

    def __init__(
        self, include_functions: bool = True, decimal_precision: int = 10, **kwargs
    ):
        """
        Initialize the calculator tool.

        Args:
            include_functions: Whether to include math functions
            decimal_precision: Number of decimal places to round results to
            **kwargs: Additional arguments passed to BaseTool
        """
        super().__init__(
            name="Calculator",
            description="Evaluate mathematical expressions safely",
            **kwargs,
        )

        self.include_functions = include_functions
        self.decimal_precision = decimal_precision

        logger.debug("Initialized Calculator tool")

    def _run(self, expression: str) -> str:
        """
        Evaluate a mathematical expression.

        Args:
            expression: The expression to evaluate

        Returns:
            The result as a string
        """
        logger.info(f"Evaluating expression: {expression}")

        try:
            # Clean up the expression
            expression = expression.strip()

            # Parse the expression
            node = ast.parse(expression, mode="eval").body

            # Evaluate it
            result = self._eval_node(node)

            # Format the result
            if isinstance(result, float):
                # Round to the specified decimal precision
                result = round(result, self.decimal_precision)

                # Convert to int if it's a whole number
                if result == int(result):
                    result = int(result)

            logger.debug(f"Evaluation result: {result}")
            return str(result)

        except Exception as e:
            error_msg = f"Error evaluating expression: {str(e)}"
            logger.warning(error_msg)
            return error_msg

    def _eval_node(self, node: ast.AST) -> Any:
        """
        Recursively evaluate an AST node.

        Args:
            node: The AST node to evaluate

        Returns:
            The result of the evaluation

        Raises:
            ValueError: If the node type is not supported
        """
        # Constants
        if isinstance(node, ast.Constant):
            return node.value

        # Names (variables/functions)
        elif isinstance(node, ast.Name):
            if self.include_functions and node.id in self._ALLOWED_FUNCTIONS:
                return self._ALLOWED_FUNCTIONS[node.id]
            raise ValueError(f"Name '{node.id}' is not defined")

        # Binary operations (a + b, a - b, etc.)
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in self._OPERATORS:
                raise ValueError(
                    f"Unsupported binary operator: {type(node.op).__name__}"
                )

            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            return self._OPERATORS[type(node.op)](left, right)

        # Unary operations (+a, -a)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in self._OPERATORS:
                raise ValueError(
                    f"Unsupported unary operator: {type(node.op).__name__}"
                )

            operand = self._eval_node(node.operand)

            return self._OPERATORS[type(node.op)](operand)

        # Function calls
        elif isinstance(node, ast.Call):
            if not self.include_functions:
                raise ValueError("Function calls are not allowed")

            func = self._eval_node(node.func)
            args = [self._eval_node(arg) for arg in node.args]

            if func not in self._ALLOWED_FUNCTIONS.values():
                raise ValueError(f"Function '{func.__name__}' is not allowed")

            return func(*args)

        # Lists [a, b, c]
        elif isinstance(node, ast.List):
            return [self._eval_node(elt) for elt in node.elts]

        # Tuples (a, b, c)
        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt) for elt in node.elts)

        # Subscripts a[b]
        elif isinstance(node, ast.Subscript):
            value = self._eval_node(node.value)

            if isinstance(node.slice, ast.Index):
                # Python 3.8 and earlier
                index = self._eval_node(node.slice.value)
            else:
                # Python 3.9+
                index = self._eval_node(node.slice)

            return value[index]

        else:
            raise ValueError(f"Unsupported node type: {type(node).__name__}")
