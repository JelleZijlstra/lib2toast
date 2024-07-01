"""Run code after compilation using lib2toast."""

import ast
from typing import Any

from blib2to3 import pygram
from blib2to3.pgen2.grammar import Grammar

from .api import compile as compile_to_ast
from .compile import Compiler


def run(
    code: str,
    *,
    filename: str = "<string>",
    grammar: Grammar = pygram.python_grammar_soft_keywords,
    compiler: Compiler = Compiler(),
) -> dict[str, Any]:
    """Run code after compilation using lib2toast."""
    tree = compile_to_ast(code, grammar=grammar, compiler=compiler)
    assert isinstance(tree, ast.Module)
    code_object = compile(tree, filename, "exec", dont_inherit=True)
    ns: dict[str, Any] = {}
    exec(code_object, ns)
    return ns
