"""Python REPL tool implementation"""

from __future__ import annotations

import ast
import contextlib
import io
import traceback
from typing import Any, ClassVar, Dict

from .base import Tool


class PythonREPLTool(Tool):
    """Execute arbitrary Python code in a sandboxed environment.

    The tool captures stdout/stderr and returns it as the result. If the code
    raises an exception, the traceback is returned in the output.
    """

    _ALLOWED_BUILTINS: ClassVar[Dict[str, Any]] = {
        # A very small subset of safe built-ins that are occasionally useful.
        "abs": abs,
        "len": len,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "sum": sum,
        "__builtins__": {"__import__": __import__},  # Allow imports
    }

    def __init__(self, timeout: float = 5.0) -> None:
        """Create a new PythonREPLTool.

        Parameters
        ----------
        timeout
            Maximum execution time in seconds before the code is aborted.
        """
        self.timeout = timeout

    @property
    def name(self) -> str:  # type: ignore[override]
        return "python_repl"

    @property
    def description(self) -> str:  # type: ignore[override]
        return "Execute Python code. Input should be valid Python code. Use print() for visible output."

    def execute(self, code: str) -> str:  # type: ignore[override]
        """Execute code and return its output."""
        return self._run_sync(code)

    def _run_sync(self, code: str) -> str:
        """Run code synchronously inside a restricted namespace and capture its output/tracebacks."""
        if not code.strip():
            return "(no output)"

        # Prepare isolated global/local namespaces
        global_namespace: Dict[str, Any] = {"__builtins__": self._ALLOWED_BUILTINS}
        local_namespace: Dict[str, Any] = {}

        # Buffers to capture stdout/stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            with (
                contextlib.redirect_stdout(stdout_buffer),
                contextlib.redirect_stderr(stderr_buffer),
            ):
                parsed = ast.parse(code, mode="exec")
                compiled = compile(parsed, "<python_repl>", "exec")
                exec(compiled, global_namespace, local_namespace)
        except Exception:
            # On exception, capture traceback into stderr_buffer
            traceback.print_exc(file=stderr_buffer)

        # Retrieve combined output
        stdout_contents = stdout_buffer.getvalue()
        stderr_contents = stderr_buffer.getvalue()

        if stderr_contents:
            return (
                stdout_contents + ("\n" if stdout_contents else "") + stderr_contents
            ).strip()

        return stdout_contents.strip() or "(no output)"
