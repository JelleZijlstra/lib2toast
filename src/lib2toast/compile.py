"""Compile a CST to an AST."""

import ast
import re
import sys
import types
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Generic, Optional, TypeVar, Union

from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pytree import NL, Leaf, Node
from typing_extensions import TypedDict

pygram.initialize(cache_dir=None)

T = TypeVar("T")
U = TypeVar("U")

LVB = Union[Leaf, ast.Constant, tuple[Leaf, Leaf]]


class UnsupportedSyntaxError(Exception):
    """Raised when some syntax is not supported on the current Python version."""


@dataclass
class Visitor(Generic[T]):
    token_type_to_name: dict[int, str] = field(default_factory=lambda: token.tok_name)
    grammar: Grammar = pygram.python_grammar_soft_keywords

    @cached_property
    def syms(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(**self.grammar.symbol2number)

    @cached_property
    def node_type_to_name(self) -> dict[int, str]:
        return {value: key for key, value in self.grammar.symbol2number.items()}

    def get_node_name(self, node: NL) -> str:
        if node.type < 256:
            return self.token_type_to_name[node.type]
        else:
            return str(self.node_type_to_name[node.type])

    def visit(self, node: NL) -> T:
        name = self.get_node_name(node)
        method = getattr(self, f"visit_{name}", self.generic_visit)
        return method(node)

    def generic_visit(self, node: NL) -> T:
        raise NotImplementedError(f"visit_{self.get_node_name(node)}")

    def visit_typed(self, node: NL, typ: type[U]) -> U:
        result = self.visit(node)
        if not isinstance(result, typ):
            raise TypeError(f"Expected {typ}, got {result}")
        return result


class LineRange(TypedDict):
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int


def get_line_range_for_leaf(leaf: Leaf) -> LineRange:
    raw_text = leaf.value.encode("utf-8")
    num_newlines = raw_text.count(b"\n")
    if num_newlines == 0:
        end_col_offset = leaf.column + len(raw_text)
    else:
        end_col_offset = len(raw_text) - raw_text.rfind(b"\n") - 1
    return LineRange(
        lineno=leaf.lineno,
        col_offset=leaf.column,
        end_lineno=leaf.lineno + num_newlines,
        end_col_offset=end_col_offset,
    )


def get_line_range(node: NL, *, ignore_last_leaf: bool = False) -> LineRange:
    if isinstance(node, Leaf):
        return get_line_range_for_leaf(node)
    else:
        begin_range = get_line_range(node.children[0])
        last_child = node.children[-1]
        if ignore_last_leaf and isinstance(last_child, Leaf):
            end_range = get_line_range(node.children[-2])
        else:
            end_range = get_line_range(last_child, ignore_last_leaf=ignore_last_leaf)
        return unify_line_ranges(begin_range, end_range)


def _get_line_range_for_lvb(node: LVB) -> LineRange:
    if isinstance(node, Leaf):
        return get_line_range_for_leaf(node)
    elif isinstance(node, tuple):
        return _get_line_range_for_lvb(node[1])
    else:
        return get_line_range_for_ast(node)


def get_line_range_for_ast(node: Union[ast.expr, ast.stmt]) -> LineRange:
    assert node.end_lineno is not None
    assert node.end_col_offset is not None
    return LineRange(
        lineno=node.lineno,
        col_offset=node.col_offset,
        end_lineno=node.end_lineno,
        end_col_offset=node.end_col_offset,
    )


def unify_line_ranges(begin_range: LineRange, end_range: LineRange) -> LineRange:
    return LineRange(
        lineno=begin_range["lineno"],
        col_offset=begin_range["col_offset"],
        end_lineno=end_range["end_lineno"],
        end_col_offset=end_range["end_col_offset"],
    )


def literal_eval(s: str) -> object:
    """Like ast.literal_eval but supports f-strings without placeholders."""
    tree = ast.parse(s, mode="eval")
    assert isinstance(tree, ast.Expression)
    if isinstance(tree.body, ast.JoinedStr):
        return "".join(ast.literal_eval(value) for value in tree.body.values)
    else:
        return ast.literal_eval(tree)


_ASTT = TypeVar("_ASTT", bound=ast.AST)


def replace(node: _ASTT, **kwargs: Any) -> _ASTT:
    """Like copy.replace() but for AST nodes."""
    fields = dict(ast.iter_fields(node))
    attributes = {attr: getattr(node, attr) for attr in node._attributes}
    new_kwargs = {**fields, **attributes, **kwargs}
    return type(node)(**new_kwargs)


def change_type(
    node: ast.AST, new_type: Callable[..., _ASTT], /, **kwargs: Any
) -> _ASTT:
    fields = dict(ast.iter_fields(node))
    attributes = {attr: getattr(node, attr) for attr in node._attributes}
    new_kwargs = {**fields, **attributes, **kwargs}
    return new_type(**new_kwargs)


def empty_arguments() -> ast.arguments:
    return ast.arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )


def _string_prefix(leaf: Leaf) -> set[str]:
    match = re.match(r"^[A-Za-z]*", leaf.value)
    assert match, repr(leaf)
    return set(match.group())


def extract_name(node: NL) -> str:
    if isinstance(node, Leaf) and node.type == token.NAME:
        return node.value
    else:
        raise UnsupportedSyntaxError("Expected an identifier")


TOKEN_TYPE_TO_BINOP = {
    token.PLUS: ast.Add,
    token.MINUS: ast.Sub,
    token.STAR: ast.Mult,
    token.SLASH: ast.Div,
    token.AT: ast.MatMult,
    token.DOUBLESLASH: ast.FloorDiv,
    token.PERCENT: ast.Mod,
    token.DOUBLESTAR: ast.Pow,
    token.LEFTSHIFT: ast.LShift,
    token.RIGHTSHIFT: ast.RShift,
    token.AMPER: ast.BitAnd,
    token.VBAR: ast.BitOr,
    token.CIRCUMFLEX: ast.BitXor,
}
TOKEN_TYPE_TO_COMPARE_OP = {
    token.EQEQUAL: ast.Eq,
    token.NOTEQUAL: ast.NotEq,
    token.LESS: ast.Lt,
    token.GREATER: ast.Gt,
    token.LESSEQUAL: ast.LtE,
    token.GREATEREQUAL: ast.GtE,
}
NAME_TO_COMPARE_OP = {"is": ast.Is, "in": ast.In}
TOKEN_TYPE_TO_UNARY_OP = {
    token.PLUS: ast.UAdd,
    token.MINUS: ast.USub,
    token.TILDE: ast.Invert,
}
TOKEN_TYPE_TO_AUGASSIGN = {
    token.PLUSEQUAL: ast.Add,
    token.MINEQUAL: ast.Sub,
    token.STAREQUAL: ast.Mult,
    token.SLASHEQUAL: ast.Div,
    token.ATEQUAL: ast.MatMult,
    token.DOUBLESLASHEQUAL: ast.FloorDiv,
    token.PERCENTEQUAL: ast.Mod,
    token.DOUBLESTAREQUAL: ast.Pow,
    token.LEFTSHIFTEQUAL: ast.LShift,
    token.RIGHTSHIFTEQUAL: ast.RShift,
    token.AMPEREQUAL: ast.BitAnd,
    token.VBAREQUAL: ast.BitOr,
    token.CIRCUMFLEXEQUAL: ast.BitXor,
}


@dataclass
class Consumer:
    """Parsing helper for parsing complicated CST nodes."""

    children: Sequence[NL]
    index: int = 0

    def consume(self, typ: Optional[int] = None) -> Optional[NL]:
        if self.index < len(self.children) and (
            typ is None or self.children[self.index].type == typ
        ):
            node = self.children[self.index]
            self.index += 1
            return node
        else:
            return None

    def consume_name(self, name: str) -> Optional[Leaf]:
        node = self.consume(token.NAME)
        if node is not None and isinstance(node, Leaf) and node.value == name:
            return node
        else:
            return None

    def expect(self, typ: Optional[int] = None) -> NL:
        node = self.consume(typ)
        if node is None:
            raise RuntimeError(f"Expected {typ}")
        return node

    def expect_name(self, name: str) -> Leaf:
        node = self.expect(token.NAME)
        assert isinstance(node, Leaf)
        if node.value != name:
            raise RuntimeError(f"Expected {name!r}")
        return node

    def next_is(self, typ: int) -> bool:
        return self.index < len(self.children) and self.children[self.index].type == typ

    def done(self) -> bool:
        return self.index >= len(self.children)


class Compiler(Visitor[ast.AST]):
    expr_context: ast.expr_context = ast.Load()

    if sys.version_info >= (3, 10):
        _match_compiler: Optional["MatchCompiler"] = None

        def get_match_compiler(self) -> "MatchCompiler":
            if self._match_compiler is None:
                self._match_compiler = MatchCompiler(
                    token_type_to_name=self.token_type_to_name,
                    grammar=self.grammar,
                    compiler=self,
                )
            return self._match_compiler

    @contextmanager
    def set_expr_context(
        self, expr_context: ast.expr_context
    ) -> Generator[None, None, None]:
        old_expr_context = self.expr_context
        self.expr_context = expr_context
        try:
            yield
        finally:
            self.expr_context = old_expr_context

    if sys.version_info >= (3, 12):

        def visit_typevar(self, node: Node) -> ast.AST:
            consumer = Consumer(node.children)
            name = extract_name(consumer.expect(token.NAME))
            bound = None
            default = None
            if consumer.consume(token.COLON) is not None:
                bound = self.visit_typed(consumer.expect(), ast.expr)
            if consumer.consume(token.EQUAL) is not None:
                default = self.visit_typed(consumer.expect(), ast.expr)
            if sys.version_info >= (3, 13):
                return ast.TypeVar(
                    name=name,
                    bound=bound,
                    default_value=default,
                    **get_line_range(node),
                )
            else:
                if default is not None and sys.version_info < (3, 13):
                    raise UnsupportedSyntaxError("TypeVar default")
                return ast.TypeVar(name=name, bound=bound, **get_line_range(node))

        def visit_paramspec(self, node: Node) -> ast.AST:
            if sys.version_info >= (3, 13):
                if len(node.children) == 4:
                    default_value = self.visit(node.children[3])
                else:
                    default_value = None
                return ast.ParamSpec(
                    name=node.children[1].value,
                    default_value=default_value,
                    **get_line_range(node),
                )
            else:
                return ast.ParamSpec(
                    name=extract_name(node.children[1]), **get_line_range(node)
                )

        def visit_typevartuple(self, node: Node) -> ast.TypeVarTuple:
            if sys.version_info >= (3, 13):
                if len(node.children) == 4:
                    default_value = self.visit(node.children[3])
                else:
                    default_value = None
                return ast.TypeVarTuple(
                    name=node.children[1].value,
                    default_value=default_value,
                    **get_line_range(node),
                )
            else:
                return ast.TypeVarTuple(
                    name=extract_name(node.children[1]), **get_line_range(node)
                )

        def _make_type_param(self, node: NL) -> ast.type_param:
            result = self.visit(node)
            if isinstance(result, ast.type_param):
                return result
            elif isinstance(result, ast.Name):
                return ast.TypeVar(name=result.id, **get_line_range(node))
            else:
                raise UnsupportedSyntaxError("Type parameter")

        def compile_typeparams(self, node: NL) -> list[ast.type_param]:
            if isinstance(node, Leaf):
                return [ast.TypeVar(name=node.value, **get_line_range(node))]
            return [self._make_type_param(child) for child in node.children[1:-1:2]]

    else:

        def compile_typeparams(self, node: NL) -> list[ast.expr]:
            raise UnsupportedSyntaxError(f"Type parameters: {node!r}")

    def compile_statement_list(self, nodes: Sequence[NL]) -> list[ast.stmt]:
        statements = []
        for node in nodes:
            if node.type in (token.ENDMARKER, token.NEWLINE):
                continue
            if isinstance(node, Node) and node.type == self.syms.simple_stmt:
                statements += self.compile_simple_stmt(node)
            else:
                statements.append(self.visit_typed(node, ast.stmt))
        return statements

    def visit_file_input(self, node: Node) -> ast.AST:
        return ast.Module(
            body=self.compile_statement_list(node.children), type_ignores=[]  # TODO
        )

    # Statements
    def compile_simple_stmt(self, node: Node) -> list[ast.stmt]:
        statements: list[ast.stmt] = []
        for child in node.children[::2]:
            val = self.visit(child)
            if isinstance(val, ast.expr):
                statements.append(ast.Expr(val, **get_line_range(child)))
            else:
                assert isinstance(val, ast.stmt)
                statements.append(val)
        return statements

    def visit_expr_stmt(self, node: Node) -> ast.AST:
        consumer = Consumer(node.children)
        lhs_node = consumer.expect()
        with self.set_expr_context(ast.Store()):
            lhs = self.visit_typed(lhs_node, ast.expr)
        if (annassign := consumer.consume(self.syms.annassign)) is not None:
            annotation = self.visit_typed(annassign.children[1], ast.expr)
            if len(annassign.children) == 4:
                value = self.visit_typed(annassign.children[3], ast.expr)
            else:
                value = None
            simple = 1 if lhs_node.type == token.NAME else 0
            if not isinstance(lhs, (ast.Name, ast.Attribute, ast.Subscript)):
                raise UnsupportedSyntaxError("AnnAssign target")
            return ast.AnnAssign(
                target=lhs,
                annotation=annotation,
                value=value,
                simple=simple,
                **get_line_range(node),
            )
        elif consumer.consume(token.EQUAL) is not None:
            exprs = []
            rhs: Optional[ast.expr] = None
            while True:
                child = consumer.expect()
                if consumer.index < len(node.children):
                    with self.set_expr_context(ast.Store()):
                        exprs.append(self.visit_typed(child, ast.expr))
                else:
                    rhs = self.visit_typed(child, ast.expr)
                if consumer.done():
                    break
                else:
                    consumer.expect(token.EQUAL)
            assert rhs is not None
            return ast.Assign(targets=[lhs, *exprs], value=rhs, **get_line_range(node))
        else:
            # AugAssign
            operator = consumer.expect()
            assert isinstance(operator, Leaf)
            op = TOKEN_TYPE_TO_AUGASSIGN[operator.type]()
            if not isinstance(lhs, (ast.Name, ast.Attribute, ast.Subscript)):
                raise UnsupportedSyntaxError("AugAssign target")
            return ast.AugAssign(
                target=lhs,
                op=op,
                value=self.visit_typed(consumer.expect(), ast.expr),
                **get_line_range(node),
            )

    def visit_del_stmt(self, node: Node) -> ast.AST:
        with self.set_expr_context(ast.Del()):
            target = self.visit_typed(node.children[1], ast.expr)
        if isinstance(target, ast.Tuple) and node.children[1].type != self.syms.atom:
            targets = target.elts
        else:
            targets = [target]
        return ast.Delete(targets=targets, **get_line_range(node))

    def visit_raise_stmt(self, node: Node) -> ast.Raise:
        exc = self.visit_typed(node.children[1], ast.expr)
        if len(node.children) > 2:
            assert isinstance(node.children[2], Leaf)
            if node.children[2].value != "from":
                # blib2to3 supports Python 2-style raise a, b, c
                raise UnsupportedSyntaxError("raise")
            cause = self.visit_typed(node.children[3], ast.expr)
        else:
            cause = None
        return ast.Raise(exc=exc, cause=cause, **get_line_range(node))

    def visit_assert_stmt(self, node: Node) -> ast.Assert:
        return ast.Assert(
            test=self.visit_typed(node.children[1], ast.expr),
            msg=(
                self.visit_typed(node.children[3], ast.expr)
                if len(node.children) > 3
                else None
            ),
            **get_line_range(node),
        )

    def visit_global_stmt(self, node: Node) -> Union[ast.Global, ast.Nonlocal]:
        names = []
        for name_node in node.children[1::2]:
            assert isinstance(name_node, Leaf)
            names.append(name_node.value)
        assert isinstance(node.children[0], Leaf)
        if node.children[0].value == "global":
            return ast.Global(names=names, **get_line_range(node))
        else:
            return ast.Nonlocal(names=names, **get_line_range(node))

    def visit_return_stmt(self, node: Node) -> ast.AST:
        return ast.Return(
            value=self.visit_typed(node.children[1], ast.expr), **get_line_range(node)
        )

    def visit_import_name(self, node: Node) -> ast.Import:
        aliases = self.compile_aliases(node.children[1])
        return ast.Import(names=aliases, **get_line_range(node))

    def compile_aliases(self, node: NL) -> list[ast.alias]:
        aliases: list[ast.alias] = []
        if isinstance(node, Node) and node.type in (
            self.syms.dotted_as_names,
            self.syms.import_as_names,
        ):
            children = node.children[::2]
        else:
            children = [node]
        for child in children:
            if child.type in (self.syms.dotted_as_name, self.syms.import_as_name):
                name_node, _, asname_node = child.children
                asname = extract_name(asname_node)
            else:
                name_node = child
                asname = None
            if isinstance(name_node, Leaf):
                name = extract_name(name_node)
            else:
                name_pieces = []
                for c in name_node.children:
                    assert isinstance(c, Leaf) and c.type in (token.NAME, token.DOT)
                    name_pieces.append(c.value)
                name = "".join(name_pieces)

            aliases.append(ast.alias(name=name, asname=asname, **get_line_range(child)))
        return aliases

    def _resolve_dotted_name(self, node: NL) -> str:
        if isinstance(node, Leaf):
            return extract_name(node)
        else:
            name_pieces = []
            for c in node.children:
                assert isinstance(c, Leaf) and c.type in (token.NAME, token.DOT)
                name_pieces.append(c.value)
            return "".join(name_pieces)

    def visit_import_from(self, node: Node) -> ast.ImportFrom:
        consumer = Consumer(node.children)
        consumer.expect_name("from")
        level = 0
        while consumer.consume(token.DOT) is not None:
            level += 1
        name_node = consumer.expect()
        if isinstance(name_node, Leaf) and name_node.value == "import":
            module = None
        else:
            module = self._resolve_dotted_name(name_node)
            consumer.expect_name("import")
        if (star := consumer.consume(token.STAR)) is not None:
            aliases = [ast.alias(name="*", asname=None, **get_line_range(star))]
        else:
            consumer.consume(token.LPAR)
            aliases = self.compile_aliases(consumer.expect())
        return ast.ImportFrom(
            module=module, names=aliases, level=level, **get_line_range(node)
        )

    def visit_type_stmt(self, node: Node) -> ast.AST:
        name = ast.Name(
            id=extract_name(node.children[1]),
            ctx=ast.Store(),
            **get_line_range(node.children[1]),
        )
        value = self.visit_typed(node.children[-1], ast.expr)
        if len(node.children) == 5:
            type_params = self.compile_typeparams(node.children[2])
        else:
            type_params = []
        if sys.version_info >= (3, 12):
            return ast.TypeAlias(
                name=name, type_params=type_params, value=value, **get_line_range(node)
            )
        else:
            raise UnsupportedSyntaxError("Type alias")

    # Compound statements
    def compile_suite(self, node: NL) -> tuple[list[ast.stmt], LineRange]:
        if isinstance(node, Node) and node.type == self.syms.suite:
            statements = self.compile_statement_list(node.children[2:-1])
            return statements, get_line_range_for_ast(statements[-1])
        else:
            statements = self.compile_statement_list([node])
            return statements, get_line_range(node, ignore_last_leaf=True)

    def consume_and_compile_suite(
        self, consumer: Consumer
    ) -> tuple[list[ast.stmt], LineRange]:
        consumer.expect(token.COLON)
        return self.compile_suite(consumer.expect())

    def visit_funcdef(self, node: Node) -> ast.FunctionDef:
        consumer = Consumer(node.children)
        consumer.expect_name("def")
        name = extract_name(consumer.expect())
        if (type_params_node := consumer.consume(self.syms.typeparams)) is not None:
            type_params = self.compile_typeparams(type_params_node)
        else:
            type_params = []
        args = self.visit_typed(consumer.expect(), ast.arguments)
        if consumer.consume(token.RARROW) is not None:
            returns = self.visit_typed(consumer.expect(), ast.expr)
        else:
            returns = None
        suite, end_line_range = self.consume_and_compile_suite(consumer)
        line_range = unify_line_ranges(get_line_range(node.children[0]), end_line_range)
        if sys.version_info >= (3, 12):
            return ast.FunctionDef(
                name=name,
                args=args,
                body=suite,
                decorator_list=[],
                returns=returns,
                type_params=type_params,
                **line_range,
            )
        else:
            return ast.FunctionDef(
                decorator_list=[],
                name=name,
                args=args,
                body=suite,
                returns=returns,
                **line_range,
            )

    def visit_classdef(self, node: Node) -> ast.ClassDef:
        consumer = Consumer(node.children)
        consumer.expect_name("class")
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
        if sys.version_info >= (3, 12):
            return ast.ClassDef(
                name=name,
                bases=bases,
                keywords=keywords,
                body=suite,
                decorator_list=[],
                type_params=type_params,
                **line_range,
            )
        else:
            return ast.ClassDef(
                name=name,
                bases=bases,
                keywords=keywords,
                body=suite,
                decorator_list=[],
                **line_range,
            )

    def visit_if_stmt(self, node: Node) -> ast.If:
        consumer = Consumer(node.children)
        consumer.expect_name("if")
        test = self.visit_typed(consumer.expect(), ast.expr)
        suite, end_line_range = self.consume_and_compile_suite(consumer)
        orelse, maybe_line_range = self._compile_if_recursive(consumer)
        if maybe_line_range is not None:
            end_line_range = maybe_line_range
        line_range = unify_line_ranges(get_line_range(node.children[0]), end_line_range)
        return ast.If(test=test, body=suite, orelse=orelse, **line_range)

    def _compile_if_recursive(
        self, consumer: Consumer
    ) -> tuple[list[ast.stmt], Optional[LineRange]]:
        if consumer.done():
            return [], None
        keyword_node = consumer.expect(token.NAME)
        keyword = extract_name(keyword_node)
        if keyword == "else":
            return self.consume_and_compile_suite(consumer)
        else:
            test = self.visit_typed(consumer.expect(), ast.expr)
            suite, end_line_range = self.consume_and_compile_suite(consumer)
            orelse, maybe_line_range = self._compile_if_recursive(consumer)
            if maybe_line_range is not None:
                end_line_range = maybe_line_range
            line_range = unify_line_ranges(get_line_range(keyword_node), end_line_range)
            return [
                ast.If(test=test, body=suite, orelse=orelse, **line_range)
            ], end_line_range

    if sys.version_info >= (3, 10):

        def visit_match_stmt(self, node: Node) -> ast.Match:
            subject = self.visit_typed(node.children[1], ast.expr)
            case_nodes = node.children[5:-1]
            results = [self.compile_case_block(case_node) for case_node in case_nodes]
            cases = [case for case, _ in results]
            line_range = unify_line_ranges(
                get_line_range(node.children[0]), results[-1][1]
            )
            return ast.Match(subject=subject, cases=cases, **line_range)

        def compile_case_block(self, node: NL) -> tuple[ast.match_case, LineRange]:
            consumer = Consumer(node.children)
            consumer.expect_name("case")
            pattern_node = consumer.expect()
            compiler = self.get_match_compiler()
            pattern = compiler.visit(pattern_node)
            if (guard_node := consumer.consume(self.syms.guard)) is not None:
                guard = self.visit_typed(guard_node.children[1], ast.expr)
            else:
                guard = None
            suite, end_line_range = self.consume_and_compile_suite(consumer)
            return (
                ast.match_case(pattern=pattern, guard=guard, body=suite),
                end_line_range,
            )

    def visit_while_stmt(self, node: Node) -> ast.While:
        consumer = Consumer(node.children)
        consumer.expect_name("while")
        test = self.visit_typed(consumer.expect(), ast.expr)
        body, end_line_range = self.consume_and_compile_suite(consumer)
        if consumer.consume_name("else") is not None:
            orelse, end_line_range = self.consume_and_compile_suite(consumer)
        else:
            orelse = []
        line_range = unify_line_ranges(get_line_range(node.children[0]), end_line_range)
        return ast.While(test=test, body=body, orelse=orelse, **line_range)

    def visit_for_stmt(self, node: Node) -> ast.For:
        consumer = Consumer(node.children)
        consumer.expect_name("for")
        with self.set_expr_context(ast.Store()):
            target = self.visit_typed(consumer.expect(), ast.expr)
        consumer.expect_name("in")
        iter = self.visit_typed(consumer.expect(), ast.expr)
        body, end_line_range = self.consume_and_compile_suite(consumer)
        if consumer.consume_name("else") is not None:
            orelse, end_line_range = self.consume_and_compile_suite(consumer)
        else:
            orelse = []
        line_range = unify_line_ranges(get_line_range(node.children[0]), end_line_range)
        return ast.For(target=target, iter=iter, body=body, orelse=orelse, **line_range)

    def visit_with_stmt(self, node: Node) -> ast.With:
        consumer = Consumer(node.children)
        consumer.expect_name("with")
        with_items = self._consume_with_items_list(consumer)
        suite, end_line_range = self.consume_and_compile_suite(consumer)
        line_range = unify_line_ranges(get_line_range(node.children[0]), end_line_range)
        return ast.With(items=with_items, body=suite, **line_range)

    def _consume_with_items_list(self, consumer: Consumer) -> list[ast.withitem]:
        with_items = []
        while not consumer.done():
            with_item_node = original_with_node = consumer.expect()
            if (
                isinstance(with_item_node, Node)
                and with_item_node.type == self.syms.atom
                and with_item_node.children[0].type == token.LPAR
            ):
                with_item_node = with_item_node.children[1]
            if (
                isinstance(with_item_node, Node)
                and with_item_node.type == self.syms.testlist_gexp
                and not any(
                    child.type == self.syms.namedexpr_test
                    for child in with_item_node.children
                )
            ):
                inner_consumer = Consumer(with_item_node.children)
                with_items += self._consume_with_items_list(inner_consumer)
            else:
                if (
                    isinstance(with_item_node, Node)
                    and with_item_node.type == self.syms.asexpr_test
                ):
                    context_expr = self.visit_typed(
                        with_item_node.children[0], ast.expr
                    )
                    with self.set_expr_context(ast.Store()):
                        optional_vars = self.visit_typed(
                            with_item_node.children[2], ast.expr
                        )
                else:
                    context_expr = self.visit_typed(original_with_node, ast.expr)
                    optional_vars = None
                with_items.append(
                    ast.withitem(context_expr=context_expr, optional_vars=optional_vars)
                )
            if consumer.consume(token.COMMA) is not None:
                continue
            elif consumer.done() or consumer.next_is(token.COLON):
                break
            else:
                raise UnsupportedSyntaxError("with")
        return with_items

    def visit_async_stmt(self, node: Node) -> ast.AST:
        async_node = node.children[0]
        assert isinstance(async_node, Leaf) and async_node.type == token.ASYNC
        begin_line_range = get_line_range(async_node)
        child = self.visit(node.children[1])
        if isinstance(child, ast.With):
            return change_type(
                child,
                ast.AsyncWith,
                lineno=begin_line_range["lineno"],
                col_offset=begin_line_range["col_offset"],
            )
        elif isinstance(child, ast.For):
            return change_type(
                child,
                ast.AsyncFor,
                lineno=begin_line_range["lineno"],
                col_offset=begin_line_range["col_offset"],
            )
        elif isinstance(child, ast.FunctionDef):
            return change_type(
                child,
                ast.AsyncFunctionDef,
                lineno=begin_line_range["lineno"],
                col_offset=begin_line_range["col_offset"],
            )
        else:
            raise UnsupportedSyntaxError("async")

    visit_async_funcdef = visit_async_stmt

    def compile_decorators(self, node: NL) -> list[ast.expr]:
        if node.type == self.syms.decorator:
            return [self.visit_typed(node.children[1], ast.expr)]
        else:
            assert node.type == self.syms.decorators, repr(node)
            return [
                self.visit_typed(child.children[1], ast.expr) for child in node.children
            ]

    def visit_decorated(self, node: Node) -> ast.stmt:
        decorator_list = self.compile_decorators(node.children[0])
        stmt = self.visit_typed(node.children[1], ast.stmt)
        return replace(stmt, decorator_list=decorator_list)

    def compile_except_clause(
        self, node: Node, outer_consumer: Consumer
    ) -> tuple[ast.excepthandler, LineRange, bool]:
        consumer = Consumer(node.children)
        consumer.expect_name("except")
        if consumer.consume(token.STAR) is not None:
            is_try_star = True
        else:
            is_try_star = False
        if (type_node := consumer.consume()) is not None:
            typ = self.visit_typed(type_node, ast.expr)
        else:
            typ = None
        if consumer.consume():  # "as" or comma
            with self.set_expr_context(ast.Store()):
                name_tree = self.visit_typed(consumer.expect(), ast.expr)
            if not isinstance(name_tree, ast.Name):
                raise UnsupportedSyntaxError("ExceptHandler name")
            name = name_tree.id
        else:
            name = None

        suite, end_line_range = self.consume_and_compile_suite(outer_consumer)
        line_range = unify_line_ranges(get_line_range(node.children[0]), end_line_range)
        return (
            ast.ExceptHandler(type=typ, name=name, body=suite, **line_range),
            end_line_range,
            is_try_star,
        )

    def visit_try_stmt(self, node: Node) -> ast.stmt:
        is_try_star = False
        consumer = Consumer(node.children)
        consumer.expect_name("try")
        body, end_line_range = self.consume_and_compile_suite(consumer)
        handlers = []
        orelse = []
        finalbody = []
        while not consumer.done():
            keyword = consumer.expect()
            if isinstance(keyword, Node):
                assert keyword.type == self.syms.except_clause
                handler, end_line_range, is_try_star = self.compile_except_clause(
                    keyword, consumer
                )
                handlers.append(handler)
            else:
                assert keyword.type == token.NAME and keyword.value in (
                    "else",
                    "except",
                    "finally",
                ), repr(keyword)
                suite, end_line_range = self.consume_and_compile_suite(consumer)
                if keyword.value == "else":
                    orelse = suite
                elif keyword.value == "except":
                    line_range = unify_line_ranges(
                        get_line_range(keyword), end_line_range
                    )
                    handlers.append(
                        ast.ExceptHandler(
                            type=None, name=None, body=suite, **line_range
                        )
                    )
                else:
                    finalbody = suite

        line_range = unify_line_ranges(get_line_range(node.children[0]), end_line_range)
        if is_try_star:
            if sys.version_info >= (3, 11):
                return ast.TryStar(
                    body=body,
                    handlers=handlers,
                    orelse=orelse,
                    finalbody=finalbody,
                    **line_range,
                )
            else:
                raise UnsupportedSyntaxError("try star")
        else:
            return ast.Try(
                body=body,
                handlers=handlers,
                orelse=orelse,
                finalbody=finalbody,
                **line_range,
            )

    # Expressions
    def visit_exprlist(self, node: Node, parent_node: Optional[Node] = None) -> ast.AST:
        if parent_node is None:
            parent_node = node
        if node.children[1].type == self.syms.old_comp_for:
            elt = self.visit_typed(node.children[0], ast.expr)
            comps = self._compile_comprehension(node.children[1])
            return ast.GeneratorExp(
                elt=elt, generators=comps, **get_line_range(parent_node)
            )
        elts = [self.visit_typed(child, ast.expr) for child in node.children[::2]]
        return ast.Tuple(
            elts=elts, ctx=self.expr_context, **get_line_range(parent_node)
        )

    visit_testlist_star_expr = visit_testlist_gexp = visit_subject_expr = visit_exprlist

    def visit_atom(self, node: Node) -> ast.AST:
        if node.children[0].type == token.LPAR:
            if len(node.children) == 2:
                return ast.Tuple(elts=[], ctx=self.expr_context, **get_line_range(node))
            # tuples, parenthesized expressions
            middle = node.children[1]
            if isinstance(middle, Node) and middle.type == self.syms.testlist_gexp:
                return self.visit_testlist_gexp(middle, node)
            return self.visit(middle)
        elif node.children[0].type == token.LSQB:
            if len(node.children) == 2:
                return ast.List(elts=[], ctx=self.expr_context, **get_line_range(node))
            # lists
            inner = node.children[1]
            if inner.type != self.syms.listmaker:
                return ast.List(
                    elts=[self.visit_typed(inner, ast.expr)],
                    ctx=self.expr_context,
                    **get_line_range(node),
                )
            if inner.children[1].type == self.syms.old_comp_for:
                elt = self.visit_typed(inner.children[0], ast.expr)
                comps = self._compile_comprehension(inner.children[1])
                return ast.ListComp(elt=elt, generators=comps, **get_line_range(node))
            elts = [self.visit_typed(child, ast.expr) for child in inner.children[::2]]
            return ast.List(elts=elts, ctx=self.expr_context, **get_line_range(node))
        elif node.children[0].type == token.LBRACE:
            if len(node.children) == 2:
                return ast.Dict(keys=[], values=[], **get_line_range(node))
            # sets, dicts
            inner = node.children[1]
            if inner.type != self.syms.dictsetmaker:
                return ast.Set(
                    elts=[self.visit_typed(inner, ast.expr)], **get_line_range(node)
                )
            consumer = Consumer(inner.children)
            is_dict = False
            keys: list[Optional[ast.expr]] = []
            values = []
            elts = []
            while not consumer.done():
                if consumer.consume(token.DOUBLESTAR) is not None:
                    is_dict = True
                    expr = consumer.expect()
                    keys.append(None)
                    values.append(self.visit_typed(expr, ast.expr))
                elif star_expr := consumer.consume(self.syms.star_expr):
                    elts.append(
                        ast.Starred(
                            value=self.visit_typed(star_expr.children[1], ast.expr),
                            ctx=self.expr_context,
                            **get_line_range(star_expr),
                        )
                    )
                else:
                    key_node = consumer.expect()
                    if (walrus := consumer.consume(token.COLONEQUAL)) is not None:
                        value_node = consumer.expect()
                        elt = self._compile_named_expr((key_node, walrus, value_node))
                        elts.append(elt)
                    elif consumer.consume(token.COLON) is not None:
                        key = self.visit_typed(key_node, ast.expr)
                        value = self.visit_typed(consumer.expect(), ast.expr)
                        keys.append(key)
                        values.append(value)
                        is_dict = True
                    else:
                        elts.append(self.visit_typed(key_node, ast.expr))
                    if comp_for := consumer.consume(self.syms.comp_for):
                        comps = self._compile_comprehension(comp_for)
                        if is_dict:
                            assert len(keys) == 1 and keys[0] is not None, keys
                            assert len(values) == 1, values
                            assert not elts, elts
                            return ast.DictComp(
                                key=keys[0],
                                value=values[0],
                                generators=comps,
                                **get_line_range(node),
                            )
                        else:
                            assert len(elts) == 1, elts
                            assert not keys, keys
                            assert not values, values
                            return ast.SetComp(
                                elt=elts[0], generators=comps, **get_line_range(node)
                            )
                if not consumer.done():
                    comma = consumer.consume(token.COMMA)
                    assert comma is not None
            if is_dict:
                assert not elts, elts
                return ast.Dict(keys=keys, values=values, **get_line_range(node))
            else:
                assert not keys, keys
                assert not values, values
                return ast.Set(elts=elts, **get_line_range(node))
        elif node.children[0].type == token.DOT:
            # ellipsis
            assert len(node.children) == 3
            assert all(child.type == token.DOT for child in node.children)
            return ast.Constant(value=Ellipsis, **get_line_range(node))
        elif node.children[0].type == token.BACKQUOTE:
            # repr. Why not support it?
            line_range = get_line_range(node)
            inner = self.visit_typed(node.children[1], ast.expr)
            return ast.JoinedStr(
                values=[
                    ast.FormattedValue(
                        value=inner, conversion=ord("r"), format_spec=None, **line_range
                    )
                ],
                **line_range,
            )
        else:
            # concatenated strings
            values = []
            last_value_bits: list[LVB] = []
            contains_fstring = False
            is_bytestring = False
            for child in node.children:
                if isinstance(child, Leaf) and child.type == token.STRING:
                    if {"b", "B"} & _string_prefix(child):
                        is_bytestring = True
                    last_value_bits.append(child)
                elif isinstance(child, Node) and child.type == self.syms.fstring:
                    if is_bytestring:
                        raise UnsupportedSyntaxError("f-string in bytestring")
                    contains_fstring = True
                    new_values, last_value_bits = self._compile_fstring_innards(
                        child.children, last_value_bits, None
                    )
                    values += new_values
            if last_value_bits:
                if is_bytestring:
                    assert not values
                    bits = []
                    for leaf in last_value_bits:
                        assert isinstance(leaf, Leaf)
                        bits.append(ast.literal_eval(leaf.value))
                    return ast.Constant(value=b"".join(bits), **get_line_range(node))
                values.append(self._concatenate_joined_strings(last_value_bits))
            if len(values) == 1 and not contains_fstring:
                return values[0]
            return self._derange(
                ast.JoinedStr(values=values, **get_line_range(node)), node
            )

    def visit_fstring(self, node: Node) -> ast.expr:
        values, last_value_bits = self._compile_fstring_innards(node.children, [], None)
        if last_value_bits:
            values.append(self._concatenate_joined_strings(last_value_bits))
        joined = ast.JoinedStr(values=values, **get_line_range(node))
        return self._derange(joined, node)

    def _derange(self, value: ast.expr, node: Node) -> ast.expr:
        if sys.version_info >= (3, 12):
            return value
        if isinstance(value, ast.Constant):
            return ast.Constant(
                value=value.value, kind=value.kind, **get_line_range(node)
            )
        elif isinstance(value, ast.FormattedValue):
            if isinstance(value.format_spec, ast.JoinedStr):
                format_spec = self._derange(value.format_spec, node)
            else:
                format_spec = value.format_spec
            return ast.FormattedValue(
                value=value.value,
                conversion=value.conversion,
                format_spec=format_spec,
                **get_line_range(node),
            )
        elif isinstance(value, ast.JoinedStr):
            return ast.JoinedStr(
                values=[self._derange(val, node) for val in value.values],
                **get_line_range(node),
            )
        else:
            return value

    def _compile_fstring_innards(
        self,
        children: Sequence[NL],
        last_value_bits: list[LVB],
        start_leaf: Optional[Leaf],
    ) -> tuple[list[ast.expr], list[LVB]]:
        values: list[ast.expr] = []
        for child in children:
            if isinstance(child, Leaf):
                if child.type == token.FSTRING_START:
                    assert start_leaf is None
                    start_leaf = child
                    continue
                elif child.type == token.FSTRING_END:
                    continue
                assert child.type == token.FSTRING_MIDDLE, repr(child)
                if child.value:
                    assert start_leaf is not None, child
                    last_value_bits.append((start_leaf, child))
            else:
                assert start_leaf is not None, children
                self_doc, formatted_value = self.compile_fstring_replacement_field(
                    child, start_leaf
                )
                if self_doc is not None:
                    last_value_bits.append(self_doc)
                if last_value_bits:
                    values.append(self._concatenate_joined_strings(last_value_bits))
                    last_value_bits = []
                values.append(formatted_value)
        return values, last_value_bits

    def compile_fstring_replacement_field(
        self, node: Node, fstring_start: Leaf
    ) -> tuple[Optional[ast.Constant], ast.FormattedValue]:
        consumer = Consumer(node.children)
        consumer.expect(token.LBRACE)
        expr_node = consumer.expect()
        expr = self.visit_typed(expr_node, ast.expr)
        self_doc = format_spec = None
        conversion = -1
        if (eq := consumer.consume(token.EQUAL)) is not None:
            text = str(expr_node) + str(eq)
            begin_line_range = get_line_range(expr_node)
            end_line_range = get_line_range(eq)
            line_range = unify_line_ranges(begin_line_range, end_line_range)
            next_node = node.children[consumer.index]
            if next_node.prefix:
                text += next_node.prefix
                raw_prefix = next_node.prefix.encode("utf-8")
                line_range["end_lineno"] += raw_prefix.count(b"\n")
                if b"\n" in raw_prefix:
                    line_range["end_col_offset"] = (
                        len(raw_prefix) - raw_prefix.rfind(b"\n") - 1
                    )
                else:
                    line_range["end_col_offset"] += len(raw_prefix)
            self_doc = ast.Constant(value=text, **line_range)
        if consumer.consume(token.BANG) is not None:
            specifier = consumer.expect(token.NAME)
            assert isinstance(specifier, Leaf)
            conversion_string = specifier.value
            if conversion_string in ("s", "r", "a"):
                conversion = ord(conversion_string)
            else:
                raise RuntimeError(f"Unexpected conversion: {conversion_string!r}")
        if (colon := consumer.consume(token.COLON)) is not None:
            values, last_value_bits = self._compile_fstring_innards(
                node.children[consumer.index : -1], [], fstring_start
            )
            if last_value_bits:
                values.append(self._concatenate_joined_strings(last_value_bits))
            elif values and sys.version_info >= (3, 12) and sys.version_info < (3, 13):
                # there's always an empty Constant for some reason
                prev_line_range = get_line_range(node.children[-2])
                next_line_range = get_line_range(node.children[-1])
                line_range = LineRange(
                    lineno=prev_line_range["end_lineno"],
                    col_offset=prev_line_range["end_col_offset"],
                    end_lineno=next_line_range["lineno"],
                    end_col_offset=next_line_range["col_offset"],
                )
                values.append(ast.Constant(value="", **line_range))
            line_range = unify_line_ranges(
                get_line_range(colon), get_line_range(node.children[-2])
            )
            format_spec = ast.JoinedStr(values=values, **line_range)
        if conversion == -1 and format_spec is None and self_doc is not None:
            conversion = ord("r")

        return self_doc, ast.FormattedValue(
            value=expr,
            conversion=conversion,
            format_spec=format_spec,
            **get_line_range(node),
        )

    def _concatenate_joined_strings(self, nodes: Sequence[LVB]) -> ast.Constant:
        strings = []
        kind = None
        for i, node in enumerate(nodes):
            if isinstance(node, ast.Constant):
                strings.append(node.value)
            elif isinstance(node, tuple):
                fstring_start, fstring_middle = node
                end = fstring_start.value.lstrip("rRfF")
                string = f"{fstring_start.value}{fstring_middle}{end}"
                strings.append(literal_eval(string))
            elif node.type == token.STRING:
                if i == 0:
                    prefix = _string_prefix(node)
                    if "u" in prefix:
                        kind = "u"
                strings.append(ast.literal_eval(node.value))
            else:
                raise RuntimeError(f"Unexpected node: {node!r}")
        line_range = unify_line_ranges(
            _get_line_range_for_lvb(nodes[0]), _get_line_range_for_lvb(nodes[-1])
        )
        return ast.Constant(value="".join(strings), kind=kind, **line_range)

    def visit_expr(self, node: Node) -> ast.expr:
        op = self.visit_typed(node.children[0], ast.expr)
        begin_range = get_line_range(node.children[0])
        for child_index in range(2, len(node.children), 2):
            child = node.children[child_index]
            operator = node.children[child_index - 1]
            op = ast.BinOp(
                left=op,
                op=TOKEN_TYPE_TO_BINOP[operator.type](),
                right=self.visit_typed(child, ast.expr),
                **unify_line_ranges(begin_range, get_line_range(child)),
            )
        return op

    visit_xor_expr = visit_and_expr = visit_shift_expr = visit_arith_expr = (
        visit_term
    ) = visit_expr

    def visit_comparison(self, node: Node) -> ast.AST:
        left = self.visit_typed(node.children[0], ast.expr)
        ops: list[ast.cmpop] = []
        comparators: list[ast.expr] = []
        for child_index in range(2, len(node.children), 2):
            child = node.children[child_index]
            operator_node = node.children[child_index - 1]
            if isinstance(operator_node, Leaf):
                if operator_node.type == token.NAME:
                    operator = NAME_TO_COMPARE_OP[operator_node.value]()
                else:
                    operator = TOKEN_TYPE_TO_COMPARE_OP[operator_node.type]()
            else:
                # is not, not in
                first, second = operator_node.children
                assert isinstance(first, Leaf) and first.type == token.NAME
                assert isinstance(second, Leaf) and second.type == token.NAME
                if first.value == "not":
                    assert second.value == "in"
                    operator = ast.NotIn()
                else:
                    assert first.value == "is"
                    assert second.value == "not"
                    operator = ast.IsNot()
            ops.append(operator)
            right = self.visit_typed(child, ast.expr)
            comparators.append(right)
        return ast.Compare(
            left=left, ops=ops, comparators=comparators, **get_line_range(node)
        )

    def visit_star_expr(self, node: Node) -> ast.AST:
        return ast.Starred(
            value=self.visit_typed(node.children[1], ast.expr),
            ctx=self.expr_context,
            **get_line_range(node),
        )

    def visit_not_test(self, node: Node) -> ast.AST:
        return ast.UnaryOp(
            op=ast.Not(),
            operand=self.visit_typed(node.children[1], ast.expr),
            **get_line_range(node),
        )

    def visit_and_test(self, node: Node) -> ast.AST:
        operands = [self.visit_typed(child, ast.expr) for child in node.children[::2]]
        return ast.BoolOp(op=ast.And(), values=operands, **get_line_range(node))

    def visit_or_test(self, node: Node) -> ast.AST:
        operands = [self.visit_typed(child, ast.expr) for child in node.children[::2]]
        return ast.BoolOp(op=ast.Or(), values=operands, **get_line_range(node))

    def visit_test(self, node: Node) -> ast.AST:
        # must be if-else
        assert len(node.children) == 5
        assert isinstance(node.children[1], Leaf) and node.children[1].value == "if"
        assert isinstance(node.children[3], Leaf) and node.children[3].value == "else"
        return ast.IfExp(
            test=self.visit_typed(node.children[2], ast.expr),
            body=self.visit_typed(node.children[0], ast.expr),
            orelse=self.visit_typed(node.children[4], ast.expr),
            **get_line_range(node),
        )

    def visit_factor(self, node: Node) -> ast.UnaryOp:
        return ast.UnaryOp(
            op=TOKEN_TYPE_TO_UNARY_OP[node.children[0].type](),
            operand=self.visit_typed(node.children[1], ast.expr),
            **get_line_range(node),
        )

    def visit_power(self, node: Node) -> ast.expr:
        children = node.children
        if len(children) > 2 and node.children[-2].type == token.DOUBLESTAR:
            operand = self.visit_typed(children[-1], ast.expr)
            return ast.BinOp(
                left=self._visit_power_without_power(children[:-2]),
                op=ast.Pow(),
                right=operand,
                **get_line_range(node),
            )
        else:
            return self._visit_power_without_power(children)

    def _visit_power_without_power(self, children: Sequence[NL]) -> ast.expr:
        if children[0].type == token.AWAIT:
            line_range = unify_line_ranges(
                get_line_range(children[0]), get_line_range(children[-1])
            )
            return ast.Await(
                value=self.compile_power_without_await(children[1:]), **line_range
            )
        else:
            return self.compile_power_without_await(children)

    def compile_power_without_await(self, children: Sequence[NL]) -> ast.expr:
        trailers = children[1:]
        if trailers:
            with self.set_expr_context(ast.Load()):
                atom = self.visit_typed(children[0], ast.expr)
        else:
            atom = self.visit_typed(children[0], ast.expr)
        begin_range = get_line_range(children[0])
        for trailer in trailers:
            assert isinstance(trailer, Node)
            if trailer is trailers[-1]:
                ctx = self.expr_context
            else:
                ctx = ast.Load()
            if trailer.children[0].type == token.LPAR:  # call
                if len(trailer.children) == 2:
                    args: list[ast.expr] = []
                    keywords: list[ast.keyword] = []
                else:
                    args, keywords = self._compile_arglist(trailer.children[1], trailer)
                atom = ast.Call(
                    func=atom,
                    args=args,
                    keywords=keywords,
                    **unify_line_ranges(
                        begin_range, get_line_range(trailer.children[-1])
                    ),
                )
            elif trailer.children[0].type == token.LSQB:  # subscript
                line_range = unify_line_ranges(
                    begin_range, get_line_range(trailer.children[-1])
                )
                with self.set_expr_context(ast.Load()):
                    subscript = self.visit_typed(trailer.children[1], ast.expr)
                if isinstance(subscript, ast.Starred):
                    subscript = ast.Tuple(
                        elts=[subscript],
                        ctx=ast.Load(),
                        **get_line_range(trailer.children[1]),
                    )
                atom = ast.Subscript(value=atom, slice=subscript, ctx=ctx, **line_range)
            elif trailer.children[0].type == token.DOT:  # attribute
                assert (
                    isinstance(trailer.children[1], Leaf)
                    and trailer.children[1].type == token.NAME
                )
                atom = ast.Attribute(
                    value=atom,
                    attr=trailer.children[1].value,
                    ctx=ctx,
                    **unify_line_ranges(
                        begin_range, get_line_range(trailer.children[1])
                    ),
                )
            else:
                raise NotImplementedError(repr(trailer))
        return atom

    def _compile_arglist(
        self, node: NL, parent_node: NL
    ) -> tuple[list[ast.expr], list[ast.keyword]]:
        if not isinstance(node, Node) or node.type != self.syms.arglist:
            arguments = [node]
        else:
            arguments = node.children[::2]
        args: list[ast.expr] = []
        keywords: list[ast.keyword] = []
        for argument in arguments:
            if isinstance(argument, Leaf) or argument.type != self.syms.argument:
                args.append(self.visit_typed(argument, ast.expr))
            elif argument.children[0].type == token.STAR:
                args.append(
                    ast.Starred(
                        value=self.visit_typed(argument.children[1], ast.expr),
                        ctx=ast.Load(),
                        **get_line_range(argument),
                    )
                )
            elif argument.children[0].type == token.DOUBLESTAR:
                keywords.append(
                    ast.keyword(
                        arg=None,
                        value=self.visit_typed(argument.children[1], ast.expr),
                        **get_line_range(argument),
                    )
                )
            elif len(argument.children) == 2:
                inner = self.visit_typed(argument.children[0], ast.expr)
                comps = self._compile_comprehension(argument.children[1])
                args.append(
                    ast.GeneratorExp(
                        elt=inner, generators=comps, **get_line_range(parent_node)
                    )
                )
            elif argument.children[1].type == token.COLONEQUAL:
                walrus = self._compile_named_expr(argument.children)
                if len(argument.children) > 3:
                    comps = self._compile_comprehension(node.children[3])
                    args.append(
                        ast.GeneratorExp(
                            elt=walrus, generators=comps, **get_line_range(parent_node)
                        )
                    )
                else:
                    args.append(walrus)
            elif argument.children[1].type == token.EQUAL:
                with self.set_expr_context(ast.Store()):
                    target = self.visit(argument.children[0])
                if not isinstance(target, ast.Name):
                    raise UnsupportedSyntaxError(
                        "keyword argument target must be a name"
                    )
                value = self.visit_typed(argument.children[2], ast.expr)
                keywords.append(
                    ast.keyword(arg=target.id, value=value, **get_line_range(argument))
                )
            else:
                raise NotImplementedError(repr(argument))
        return args, keywords

    def _compile_comprehension(self, node: NL) -> list[ast.comprehension]:
        assert node.type in (self.syms.old_comp_for, self.syms.comp_for), repr(node)
        if node.children[0].type == token.ASYNC:
            is_async = 1
            children = node.children[1:]
        else:
            is_async = 0
            children = node.children
        if len(children) > 4:
            assert isinstance(children[4], Node)
            ifs, comps = self._compile_comp_iter(children[4])
        else:
            ifs = []
            comps = []
        with self.set_expr_context(ast.Store()):
            target = self.visit_typed(children[1], ast.expr)
        comp = ast.comprehension(
            target=target,
            iter=self.visit_typed(children[3], ast.expr),
            ifs=ifs,
            is_async=is_async,
        )
        return [comp, *comps]

    def _compile_comp_iter(
        self, node: Node
    ) -> tuple[list[ast.expr], list[ast.comprehension]]:
        keyword = extract_name(node.children[0])
        if keyword == "if":
            test = self.visit(node.children[1])
            assert isinstance(test, ast.expr)
            if len(node.children) > 2:
                assert isinstance(node.children[2], Node)
                ifs, comps = self._compile_comp_iter(node.children[2])
            else:
                ifs = []
                comps = []
            return [test, *ifs], comps
        else:
            return [], self._compile_comprehension(node)

    def _compile_named_expr(self, children: Sequence[NL]) -> ast.NamedExpr:
        with self.set_expr_context(ast.Store()):
            target = self.visit(children[0])
        if not isinstance(target, ast.Name):
            raise UnsupportedSyntaxError("walrus target must be a name")
        value = self.visit_typed(children[2], ast.expr)
        return ast.NamedExpr(
            target=target,
            value=value,
            **unify_line_ranges(
                get_line_range(children[0]), get_line_range(children[2])
            ),
        )

    def visit_namedexpr_test(self, node: Node) -> ast.NamedExpr:
        return self._compile_named_expr(node.children)

    def visit_subscript(self, node: Node) -> Union[ast.expr, ast.Slice]:
        consumer = Consumer(node.children)
        if consumer.consume(token.COLON) is None:
            lower = self.visit_typed(consumer.expect(), ast.expr)
            if consumer.consume(token.COLONEQUAL) is not None:
                return self._compile_named_expr(node.children)
            assert consumer.consume(token.COLON) is not None
        else:
            lower = None
        if (sliceop := consumer.consume(self.syms.sliceop)) is not None:
            step = self.visit_typed(sliceop.children[1], ast.expr)
            upper = None
        elif consumer.consume(token.COLON) is None:
            if (upper_node := consumer.consume()) is None:
                upper = None
            else:
                upper = self.visit_typed(upper_node, ast.expr)
            if (sliceop := consumer.consume(self.syms.sliceop)) is not None:
                step = self.visit_typed(sliceop.children[1], ast.expr)
            else:
                step = None
        else:
            upper = step = None
        return ast.Slice(lower=lower, upper=upper, step=step, **get_line_range(node))

    def visit_subscriptlist(self, node: Node) -> ast.Tuple:
        elts = [self.visit_typed(child, ast.expr) for child in node.children[::2]]
        return ast.Tuple(elts=elts, ctx=self.expr_context, **get_line_range(node))

    def visit_lambdef(self, node: Node) -> ast.Lambda:
        maybe_args = node.children[1]
        if maybe_args.type == token.COLON:
            args = empty_arguments()
        else:
            args = self.compile_args(maybe_args)
        body = self.visit_typed(node.children[-1], ast.expr)
        return ast.Lambda(args=args, body=body, **get_line_range(node))

    def visit_parameters(self, node: Node) -> ast.arguments:
        if len(node.children) == 2:
            return empty_arguments()
        return self.compile_args(node.children[1])

    def compile_args(self, node: NL) -> ast.arguments:
        if (isinstance(node, Leaf) and node.type == token.NAME) or (
            isinstance(node, Node)
            and node.type in (self.syms.tname, self.syms.tname_star)
        ):
            children = [node]
        else:
            children = node.children
        posonlyargs: list[ast.arg] = []
        args: list[ast.arg] = []
        vararg = None
        kwonlyargs: list[ast.arg] = []
        kw_defaults: list[Optional[ast.expr]] = []
        kwarg = None
        defaults: list[ast.expr] = []
        current_args = args

        consumer = Consumer(children)
        while True:
            tok = consumer.consume()
            if tok is None:
                break
            if isinstance(tok, Leaf) and tok.type == token.NAME:
                current_args.append(
                    ast.arg(
                        arg=tok.value,
                        annotation=None,
                        type_comment=None,
                        **get_line_range(tok),
                    )
                )
                if consumer.consume(token.EQUAL) is not None:
                    default = self.visit_typed(consumer.expect(), ast.expr)
                    if current_args is kwonlyargs:
                        kw_defaults.append(default)
                    else:
                        defaults.append(default)
                elif current_args is kwonlyargs:
                    kw_defaults.append(None)
            elif isinstance(tok, Node) and tok.type in (
                self.syms.tname,
                self.syms.tname_star,
            ):
                arg_name = extract_name(tok.children[0])
                annotation = self.visit_typed(tok.children[2], ast.expr)
                current_args.append(
                    ast.arg(
                        arg=arg_name,
                        annotation=annotation,
                        type_comment=None,
                        **get_line_range(tok),
                    )
                )
                if consumer.consume(token.EQUAL) is not None:
                    default = self.visit_typed(consumer.expect(), ast.expr)
                    if current_args is kwonlyargs:
                        kw_defaults.append(default)
                    else:
                        defaults.append(default)
                elif current_args is kwonlyargs:
                    kw_defaults.append(None)
            elif tok.type == token.SLASH:
                posonlyargs = current_args
                current_args = args = []
            elif tok.type == token.DOUBLESTAR:
                if kwarg is not None:
                    raise UnsupportedSyntaxError("Multiple **kwargs")
                args_node = consumer.expect()
                if isinstance(args_node, Leaf):
                    name = args_node.value
                    annotation = None
                else:
                    assert args_node.type == self.syms.tname
                    assert isinstance(args_node.children[0], Leaf)
                    name = args_node.children[0].value
                    annotation = self.visit_typed(args_node.children[2], ast.expr)
                kwarg = ast.arg(
                    arg=name,
                    annotation=annotation,
                    type_comment=None,
                    **get_line_range(args_node),
                )
            elif tok.type == token.STAR:
                if node.children[consumer.index].type == token.COMMA:
                    # kw-only marker
                    pass
                else:
                    # *args
                    if vararg is not None:
                        raise UnsupportedSyntaxError("Multiple *args")
                    args_node = consumer.expect()
                    if isinstance(args_node, Leaf):
                        name = args_node.value
                        annotation = None
                    else:
                        assert args_node.type == self.syms.tname_star
                        assert isinstance(args_node.children[0], Leaf)
                        name = args_node.children[0].value
                        annotation = self.visit_typed(args_node.children[2], ast.expr)
                    vararg = ast.arg(
                        arg=name,
                        annotation=annotation,
                        type_comment=None,
                        **get_line_range(args_node),
                    )
                current_args = kwonlyargs
            else:
                raise UnsupportedSyntaxError(f"Unexpected token: {tok!r}")
            comma = consumer.consume(token.COMMA)
            if comma is None:
                assert consumer.done()
        return ast.arguments(
            posonlyargs=posonlyargs,
            args=args,
            vararg=vararg,
            kwonlyargs=kwonlyargs,
            kw_defaults=kw_defaults,
            kwarg=kwarg,
            defaults=defaults,
        )

    def visit_yield_expr(self, node: Node) -> Union[ast.Yield, ast.YieldFrom]:
        if node.children[1].type == self.syms.yield_arg:
            return ast.YieldFrom(
                value=self.visit_typed(node.children[1].children[1], ast.expr),
                **get_line_range(node),
            )
        return ast.Yield(
            value=self.visit_typed(node.children[1], ast.expr), **get_line_range(node)
        )

    # Leaves
    def visit_NAME(self, leaf: Leaf) -> ast.AST:
        line_range = get_line_range(leaf)
        if leaf.value == "return":
            return ast.Return(value=None, **line_range)
        elif leaf.value == "pass":
            return ast.Pass(**line_range)
        elif leaf.value == "break":
            return ast.Break(**line_range)
        elif leaf.value == "continue":
            return ast.Continue(**line_range)
        elif leaf.value == "yield":
            return ast.Yield(value=None, **line_range)
        elif leaf.value == "raise":
            return ast.Raise(exc=None, cause=None, **line_range)
        elif leaf.value == "None":
            return ast.Constant(value=None, **line_range)
        elif leaf.value == "True":
            return ast.Constant(value=True, **line_range)
        elif leaf.value == "False":
            return ast.Constant(value=False, **line_range)
        return ast.Name(id=leaf.value, ctx=self.expr_context, **line_range)

    def visit_NUMBER(self, leaf: Leaf) -> ast.Constant:
        return ast.Constant(value=ast.literal_eval(leaf.value), **get_line_range(leaf))

    def visit_STRING(self, leaf: Leaf) -> ast.Constant:
        prefix = _string_prefix(leaf)
        kind = "u" if "u" in prefix else None
        return ast.Constant(
            value=ast.literal_eval(leaf.value), kind=kind, **get_line_range(leaf)
        )

    def visit_COLON(self, leaf: Leaf) -> ast.Slice:
        return ast.Slice(lower=None, upper=None, step=None, **get_line_range(leaf))

    def visit_ENDMARKER(self, _leaf: Leaf) -> ast.Module:
        # empty module
        return ast.Module(body=[], type_ignores=[])


if sys.version_info >= (3, 10):

    @dataclass(kw_only=True)
    class MatchCompiler(Visitor[ast.pattern]):
        compiler: Compiler

        def visit_NAME(self, leaf: Leaf) -> ast.pattern:
            if leaf.value == "_":
                return ast.MatchAs(pattern=None, name=None, **get_line_range(leaf))
            elif leaf.value == "None":
                return ast.MatchSingleton(value=None, **get_line_range(leaf))
            elif leaf.value == "True":
                return ast.MatchSingleton(value=True, **get_line_range(leaf))
            elif leaf.value == "False":
                return ast.MatchSingleton(value=False, **get_line_range(leaf))
            return ast.MatchAs(pattern=None, name=leaf.value, **get_line_range(leaf))

        def visit_STRING(self, leaf: Leaf) -> ast.pattern:
            expr = self.compiler.visit_STRING(leaf)
            return ast.MatchValue(value=expr, **get_line_range(leaf))

        visit_NUMBER = visit_STRING

        def visit_pattern(self, node: Node) -> ast.pattern:
            pattern = self.visit_typed(node.children[0], ast.pattern)
            name = extract_name(node.children[2])
            return ast.MatchAs(pattern=pattern, name=name, **get_line_range(node))

        def visit_patterns(self, node: Node) -> ast.pattern:
            patterns = [
                self.visit_typed(child, ast.pattern) for child in node.children[::2]
            ]
            return ast.MatchSequence(patterns=patterns, **get_line_range(node))

        def visit_testlist_gexp(
            self, node: Node, parent_node: Optional[Node] = None
        ) -> ast.MatchSequence:
            if parent_node is None:
                parent_node = node
            if node.children[1].type == self.syms.old_comp_for:
                raise UnsupportedSyntaxError("comprehension in pattern matching")
            elts = [
                self.visit_typed(child, ast.pattern) for child in node.children[::2]
            ]
            return ast.MatchSequence(patterns=elts, **get_line_range(parent_node))

        def visit_expr(self, node: Node) -> ast.AST:
            for operator in node.children[1::2]:
                if operator.type != token.VBAR:
                    raise UnsupportedSyntaxError("operator in pattern matching")
            patterns = [
                self.visit_typed(child, ast.pattern) for child in node.children[::2]
            ]
            return ast.MatchOr(patterns=patterns, **get_line_range(node))

        def visit_star_expr(self, node: Node) -> ast.pattern:
            name = extract_name(node.children[1])
            if name == "_":
                return ast.MatchStar(name=None, **get_line_range(node))
            return ast.MatchStar(name=name, **get_line_range(node))

        def visit_asexpr_test(self, node: Node) -> ast.pattern:
            pattern = self.visit_typed(node.children[0], ast.pattern)
            name = extract_name(node.children[2])
            return ast.MatchAs(pattern=pattern, name=name, **get_line_range(node))

        def visit_term(self, node: Node) -> ast.pattern:
            expr = self.compiler.visit_term(node)
            return ast.MatchValue(value=expr, **get_line_range(node))

        visit_xor_expr = visit_and_expr = visit_shift_expr = visit_arith_expr = (
            visit_term
        )

        def visit_factor(self, node: Node) -> ast.pattern:
            factor = self.compiler.visit_factor(node)
            return ast.MatchValue(value=factor, **get_line_range(node))

        def visit_atom(self, node: Node) -> ast.pattern:
            if node.children[0].type == token.LPAR:
                if len(node.children) == 2:
                    return ast.MatchSequence(patterns=[], **get_line_range(node))
                # tuples, parenthesized expressions
                middle = node.children[1]
                if isinstance(middle, Node) and middle.type == self.syms.testlist_gexp:
                    return self.visit_testlist_gexp(middle, node)
                return self.visit(middle)
            elif node.children[0].type == token.LSQB:
                if len(node.children) == 2:
                    return ast.MatchSequence(patterns=[], **get_line_range(node))
                # lists
                inner = node.children[1]
                if inner.type != self.syms.listmaker:
                    return ast.MatchSequence(
                        patterns=[self.visit_typed(inner, ast.pattern)],
                        **get_line_range(node),
                    )
                if inner.children[1].type == self.syms.old_comp_for:
                    raise UnsupportedSyntaxError("comprehension in pattern matching")
                elts = [
                    self.visit_typed(child, ast.pattern)
                    for child in inner.children[::2]
                ]
                return ast.MatchSequence(patterns=elts, **get_line_range(node))
            elif node.children[0].type == token.LBRACE:
                if len(node.children) == 2:
                    return ast.MatchMapping(
                        keys=[], patterns=[], rest=None, **get_line_range(node)
                    )
                inner = node.children[1]
                if inner.type != self.syms.dictsetmaker:
                    raise UnsupportedSyntaxError("set in pattern matching")
                consumer = Consumer(inner.children)
                keys = []
                patterns = []
                rest = None
                while not consumer.done():
                    if consumer.consume(token.DOUBLESTAR) is not None:
                        rest = extract_name(consumer.expect(token.NAME))
                    elif consumer.consume(self.syms.star_expr):
                        raise UnsupportedSyntaxError(
                            "starred expression in pattern matching"
                        )
                    else:
                        key_node = consumer.expect()
                        if consumer.consume(token.COLONEQUAL) is not None:
                            raise UnsupportedSyntaxError(
                                "named expression in pattern matching"
                            )
                        elif consumer.consume(token.COLON) is not None:
                            key = self.compiler.visit_typed(key_node, ast.expr)
                            pattern = self.visit_typed(consumer.expect(), ast.pattern)
                            keys.append(key)
                            patterns.append(pattern)
                        else:
                            raise UnsupportedSyntaxError("set in pattern matching")
                        if consumer.consume(self.syms.comp_for):
                            raise UnsupportedSyntaxError(
                                "comprehension in pattern matching"
                            )
                    if not consumer.done():
                        consumer.expect(token.COMMA)
                return ast.MatchMapping(
                    keys=keys, patterns=patterns, rest=rest, **get_line_range(node)
                )
            elif node.children[0].type == token.DOT:
                # ellipsis
                raise UnsupportedSyntaxError("ellipsis in pattern matching")
            elif node.children[0].type == token.BACKQUOTE:
                raise UnsupportedSyntaxError("repr in pattern matching")
            else:
                # concatenated strings
                strings: list[Leaf] = []
                for child in node.children:
                    if isinstance(child, Leaf) and child.type == token.STRING:
                        strings.append(child)
                    else:
                        raise UnsupportedSyntaxError("f-string in pattern matching")
                string = self._concatenate_joined_strings(strings)
                return ast.MatchValue(value=string, **get_line_range(node))

        def _concatenate_joined_strings(self, nodes: Sequence[Leaf]) -> ast.Constant:
            strings = []
            for node in nodes:
                if node.type == token.STRING:
                    strings.append(ast.literal_eval(node.value))
                else:
                    raise RuntimeError(f"Unexpected node: {node!r}")
            line_range = unify_line_ranges(
                get_line_range_for_leaf(nodes[0]), get_line_range_for_leaf(nodes[-1])
            )
            return ast.Constant(value=strings[0][:0].join(strings), **line_range)

        def visit_power(self, node: Node) -> ast.pattern:
            children = node.children
            if len(children) > 2 and node.children[-2].type == token.DOUBLESTAR:
                raise UnsupportedSyntaxError("power in pattern matching")
            if children[0].type == token.AWAIT:
                raise UnsupportedSyntaxError("await in pattern matching")
            if len(children) < 2:
                raise UnsupportedSyntaxError("trailer in pattern matching")
            trailer = children[-1]
            if trailer.children[0].type == token.LPAR:  # call
                cls = self.compiler.compile_power_without_await(children[:-1])
                patterns: list[ast.pattern] = []
                kwd_attrs: list[str] = []
                kwd_patterns: list[ast.pattern] = []
                if len(trailer.children) == 2:
                    arguments = []
                else:
                    arglist = trailer.children[1]
                    if (
                        not isinstance(arglist, Node)
                        or arglist.type != self.syms.arglist
                    ):
                        arguments = [arglist]
                    else:
                        arguments = arglist.children[::2]
                for argument in arguments:
                    if (
                        isinstance(argument, Leaf)
                        or argument.type != self.syms.argument
                    ):
                        patterns.append(self.visit_typed(argument, ast.pattern))
                    elif argument.children[0].type == token.STAR:
                        raise UnsupportedSyntaxError(
                            "starred expression in pattern matching"
                        )
                    elif argument.children[0].type == token.DOUBLESTAR:
                        raise UnsupportedSyntaxError(
                            "double-starred expression in pattern matching"
                        )
                    elif len(argument.children) == 2:
                        raise UnsupportedSyntaxError(
                            "comprehension in pattern matching"
                        )
                    elif argument.children[1].type == token.COLONEQUAL:
                        raise UnsupportedSyntaxError(
                            "named expression in pattern matching"
                        )
                    elif argument.children[1].type == token.EQUAL:
                        name = extract_name(argument.children[0])
                        kwd_attrs.append(name)
                        value = self.visit_typed(argument.children[2], ast.pattern)
                        kwd_patterns.append(value)
                    elif (
                        isinstance(argument.children[1], Leaf)
                        and argument.children[1].type == token.NAME
                        and argument.children[1].value == "as"
                    ):
                        pattern_node, _, name_node = argument.children
                        pattern = self.visit_typed(pattern_node, ast.pattern)
                        name = extract_name(name_node)
                        patterns.append(
                            ast.MatchAs(
                                pattern=pattern, name=name, **get_line_range(argument)
                            )
                        )
                    else:
                        raise NotImplementedError(repr(argument))
                return ast.MatchClass(
                    cls=cls,
                    patterns=patterns,
                    kwd_attrs=kwd_attrs,
                    kwd_patterns=kwd_patterns,
                    **get_line_range(node),
                )
            elif trailer.children[0].type == token.DOT:  # subscript
                expr = self.compiler.visit_typed(node, ast.expr)
                return ast.MatchValue(value=expr, **get_line_range(node))
            else:
                raise UnsupportedSyntaxError("trailer in pattern matching")
