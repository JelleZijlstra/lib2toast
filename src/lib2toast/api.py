"""Public API functions."""

import ast
import sys

from blib2to3 import pygram
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pytree import NL

from .compile import Compiler
from .unicode_fix import fixup_unicode


def parse(code: str, grammar: Grammar = pygram.python_grammar_soft_keywords) -> NL:
    """Parse the given code using the given grammar."""
    driver = pygram.driver.Driver(grammar)
    return driver.parse_string(code, debug=True)


def compile(code: str) -> ast.AST:
    tree = parse(code + "\n")
    fixup_unicode(tree)
    return Compiler().visit(tree)


if __name__ == "__main__":
    import sys

    code = sys.argv[1]
    print(ast.dump(compile(code)))
