import ast
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pytree import NL, Node
from lib2toast.api import load_grammar, run
from lib2toast.compile import (
    Compiler,
    Consumer,
    LineRange,
    extract_name,
    get_line_range,
    replace,
    unify_line_ranges,
)

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
    compiler = BracesCompiler(grammar=grammar)
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
        compiler=compiler,
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
        compiler=compiler,
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
        compiler=compiler,
    )


class DataclassCompiler(Compiler):
    def visit_dataclassdef(self, node: Node) -> ast.stmt:
        consumer = Consumer(node.children)
        keyword_node = consumer.expect_name("dataclass")
        dataclass_bases: list[ast.expr] = []
        dataclass_keywords: list[ast.keyword] = []
        if consumer.consume(token.LPAR) is not None:
            next_node = consumer.expect()
            if next_node.type != token.RPAR:
                dataclass_bases, dataclass_keywords = self._compile_arglist(
                    next_node, next_node
                )
                consumer.expect(token.RPAR)
        name = extract_name(consumer.expect())
        if (type_params_node := consumer.consume(self.syms.typeparams)) is not None:
            type_params = self.compile_typeparams(type_params_node)
        else:
            type_params = []
        bases: list[ast.expr] = []
        keywords: list[ast.keyword] = []
        if consumer.consume(token.LPAR) is not None:
            next_node = consumer.expect()
            if next_node.type != token.RPAR:
                bases, keywords = self._compile_arglist(next_node, next_node)
                consumer.expect(token.RPAR)
        suite, end_line_range = self.consume_and_compile_suite(consumer)
        line_range = unify_line_ranges(get_line_range(node.children[0]), end_line_range)

        keyword_line_range = get_line_range(keyword_node)
        dataclass = ast.Attribute(
            value=ast.Call(
                func=ast.Name(id="__import__", ctx=ast.Load(), **keyword_line_range),
                args=[ast.Constant(value="dataclasses", **keyword_line_range)],
                keywords=[],
                **keyword_line_range,
            ),
            attr="dataclass",
            ctx=ast.Load(),
            **keyword_line_range,
        )
        decorator: ast.expr
        if dataclass_bases or dataclass_keywords:
            decorator = ast.Call(
                func=dataclass,
                args=dataclass_bases,
                keywords=dataclass_keywords,
                **keyword_line_range,
            )
        else:
            decorator = dataclass
        if sys.version_info >= (3, 12):
            return ast.ClassDef(
                name=name,
                bases=bases,
                keywords=keywords,
                body=suite,
                decorator_list=[decorator],
                type_params=type_params,
                **line_range,
            )
        else:
            return ast.ClassDef(
                name=name,
                bases=bases,
                keywords=keywords,
                body=suite,
                decorator_list=[decorator],
                **line_range,
            )

    def visit_decorated(self, node: Node) -> ast.stmt:
        decorator_list = self.compile_decorators(node.children[0])
        stmt = self.visit_typed(node.children[1], ast.stmt)
        if isinstance(stmt, ast.ClassDef):
            new_decorator_list = decorator_list + stmt.decorator_list
        else:
            new_decorator_list = decorator_list
        return replace(stmt, decorator_list=new_decorator_list)


def test_dataclass() -> None:
    dataclass_path = Path(__file__).parent / "grammars" / "dataclass.txt"
    grammar = load_grammar(dataclass_path)
    compiler = DataclassCompiler(grammar=grammar)
    check_run(
        """
        dataclass C:
            x: int
            y: int = 0
        c = C(1)
        x = c.x
        y = c.y
        """,
        {"x": 1, "y": 0},
        grammar=grammar,
        compiler=compiler,
    )
    check_run(
        """
        dataclass(frozen=True) C:
            x: int
            y: int = 0
        c = C(1)
        x = c.x
        y = c.y

        try:
            c.y = 4
        except Exception:
            caught = True
        """,
        {"x": 1, "y": 0, "caught": True},
        grammar=grammar,
        compiler=compiler,
    )
    check_run(
        """
        def deco(cls):
            cls.x = 1
            return cls

        @deco
        dataclass C:
            y: int

        c = C(2)
        x = C.x
        y = c.y
        """,
        {"x": 1, "y": 2},
        grammar=grammar,
        compiler=compiler,
    )
