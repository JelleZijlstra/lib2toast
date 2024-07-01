import ast
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pytree import NL, Node
from lib2toast.api import load_grammar
from lib2toast.compile import Compiler, Consumer, LineRange, get_line_range
from lib2toast.run import run

syms = pygram.python_symbols


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
        consumer = Consumer(node.children)
        statement_list = []
        consumer.consume(token.LBRACE)
        while not consumer.next_is(token.RBRACE):
            statement_list.append(consumer.expect())
        statements = self.compile_statement_list(statement_list)
        return statements, get_line_range(node, ignore_last_leaf=True)

    def compile_statement_list(self, nodes: Sequence[NL]) -> list[ast.stmt]:
        statements = []
        for node in nodes:
            if node.type in (token.ENDMARKER, token.NEWLINE):
                continue
            if isinstance(node, Node) and node.type == syms.simple_stmt:
                statements += self.compile_simple_stmt(node)
            else:
                stmt = self.visit(node)
                if isinstance(stmt, ast.stmt):
                    statements.append(stmt)
                elif isinstance(stmt, ast.expr):
                    statements.append(ast.Expr(stmt, **get_line_range(node)))
                else:
                    raise AssertionError(f"Unexpected statement: {stmt}")
        return statements


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
    check_run(
        """
        def f(x) {
            return x + 1
        }
        lst = []
        for i in range(4) {
            lst.append(f(i))
        }
        """,
        {"lst": [1, 2, 3, 4]},
        grammar=grammar,
        compiler=BracesCompiler(),
    )
    check_run(
        """
        class C {
            def meth(self) {
                self.x = 1
            }
            def meth2(self, x) {
                self.x = x
            }
        }
        c = C()
        c.meth()
        x1 = c.x
        c.meth2(2)
        x2 = c.x
        """,
        {"x1": 1, "x2": 2},
        grammar=grammar,
        compiler=BracesCompiler(),
    )
