import ast
import textwrap
from pathlib import Path
from typing import Any

from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.token import NL
from lib2toast.api import load_grammar
from lib2toast.compile import Compiler, Consumer, LineRange, get_line_range
from lib2toast.run import run


def check_run(
    code: str, expected_vars: dict[str, Any], grammar: Grammar, compiler: Compiler
) -> None:
    ns = run(textwrap.dedent(code), grammar=grammar, compiler=compiler)
    for key, value in expected_vars.items():
        assert ns[key] == value


class BracesCompiler(Compiler):
    def consume_and_compile_suite(
        self, consumer: Consumer
    ) -> tuple[list[ast.stmt], LineRange]:
        return self.compile_suite(consumer.expect())

    def compile_suite(self, node: NL) -> tuple[list[ast.stmt], LineRange]:
        statements = self.compile_statement_list([node.children[1]])
        return statements, get_line_range(node, ignore_last_leaf=True)


def test_braces() -> None:
    braces_path = Path(__file__).parent / "grammars" / "braces.txt"
    grammar = load_grammar(braces_path)
    check_run(
        """
        if True {
            x = 1
        }
        else {
            x = 2
        }
        """,
        {"x": 1},
        grammar=grammar,
        compiler=BracesCompiler(),
    )
