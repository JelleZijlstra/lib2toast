"""Public API functions."""

import ast
import functools
import sys
from pathlib import Path

from blib2to3 import pygram
from blib2to3.pgen2 import pgen
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pytree import NL

from .compile import Compiler
from .unicode_fix import fixup_unicode


def parse(code: str, *, grammar: Grammar = pygram.python_grammar_soft_keywords) -> NL:
    """Parse the given code using the given grammar."""
    driver = pygram.driver.Driver(grammar)
    return driver.parse_string(code, debug=True)


def compile(
    code: str,
    *,
    grammar: Grammar = pygram.python_grammar_soft_keywords,
    compiler: Compiler = Compiler(),
) -> ast.AST:
    tree = parse(code + "\n", grammar=grammar)
    fixup_unicode(tree)
    return compiler.visit(tree)


@functools.cache
def load_grammar(path: Path, *, async_keywords: bool = True) -> Grammar:
    """Load the grammar from the given path."""
    grammar = pgen.generate_grammar(path)
    grammar.async_keywords = async_keywords
    return grammar


if __name__ == "__main__":
    import sys

    code = sys.argv[1]
    print(ast.dump(compile(code)))
