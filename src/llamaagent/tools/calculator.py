"""Calculator tool implementation"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Dict, Type, Union, overload

from .base import Tool

# --------------------------------------------------------------------------- #
# Type aliases                                                                #
# --------------------------------------------------------------------------- #

Number = Union[int, float]

# A binary operator takes two numbers, a unary operator takes one number.
BinaryOp = Callable[[Number, Number], Number]
UnaryOp = Callable[[Number], Number]
OpFunc = Union[BinaryOp, UnaryOp]


# --------------------------------------------------------------------------- #
# Public tool                                                                 #
# --------------------------------------------------------------------------- #


class CalculatorTool(Tool):
    """A tiny safe calculator utility used by LlamaAgent tools/tests."""

    # ------------------------------------------------------------------ #
    # Meta information required by the Tool ABC                          #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:  # type: ignore[override]
        return "calculator"

    @property
    def description(self) -> str:  # type: ignore[override]
        return "Performs mathematical calculations safely"

    # ------------------------------------------------------------------ #
    # Supported operators                                                #
    # ------------------------------------------------------------------ #
    _ops: Dict[Type[ast.AST], OpFunc] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.BitXor: operator.xor,
    }

    # ------------------------------------------------------------------ #
    # Expression evaluation – kept synchronous and side-effect free      #
    # ------------------------------------------------------------------ #

    def _eval_expr(self, node: ast.AST) -> Number:  # noqa: C901  (small & clear)
        """Recursively evaluate node.

        Only numbers and the operators from _ops are allowed; any other
        AST node raises TypeError.
        """
        if isinstance(node, ast.Constant):  # Python 3.8+: covers Num/Str/…
            value: Any = node.value
            if not isinstance(value, (int, float)):
                raise TypeError(f"Unsupported literal: {value!r}")
            return value

        if isinstance(node, ast.BinOp):
            op_func = self._require_op(node.op)
            lhs = self._eval_expr(node.left)
            rhs = self._eval_expr(node.right)
            # mypy/pyright know the call is legal because we cast in _require_op
            return op_func(lhs, rhs)  # type: ignore[arg-type]

        if isinstance(node, ast.UnaryOp):
            op_func = self._require_op(node.op)
            operand = self._eval_expr(node.operand)
            return op_func(operand)  # type: ignore[arg-type]

        raise TypeError(f"Unsupported expression: {ast.dump(node)}")

    # ------------------------------------------------------------------ #
    # Tool interface                                                     #
    # ------------------------------------------------------------------ #

    def execute(self, expression: str) -> str:  # type: ignore[override]
        """Evaluate expression and return the result as a string.

        Any error yields a message containing the word "Error" so the upstream
        test-suite can detect failure reliably.
        """
        try:
            tree = ast.parse(expression, mode="eval")
            result = self._eval_expr(tree.body)
            return str(result)
        except Exception as exc:  # pylint: disable=broad-except
            return f"Error: {exc}"

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    @overload
    def _require_op(self, op: ast.operator) -> BinaryOp:
        ...  # noqa: D401

    @overload
    def _require_op(self, op: ast.unaryop) -> UnaryOp:
        ...  # noqa: D401

    def _require_op(self, op: Union[ast.operator, ast.unaryop]) -> OpFunc:
        """Return the function implementing op or raise TypeError."""
        op_type: Type[ast.AST] = type(op)
        if op_type in self._ops:
            return self._ops[op_type]
        raise TypeError(f"Operator {op_type.__name__} is not supported")
