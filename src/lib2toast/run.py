"""Run code after compilation using lib2toast."""

import ast
from typing import Any

from .compile import compile as compile_to_ast


def run(code: str, *, filename: str = "<string>") -> dict[str, Any]:
    """Run code after compilation using lib2toast."""
    tree = compile_to_ast(code)
    assert isinstance(tree, ast.Module)
    code_object = compile(tree, filename, "exec", dont_inherit=True)
    ns: dict[str, Any] = {}
    exec(code_object, ns)
    return ns
