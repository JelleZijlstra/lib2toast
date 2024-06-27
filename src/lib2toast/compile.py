"""Compile a CST to an AST."""

import ast
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, TypeVar

from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pytree import NL, Leaf, Node, type_repr
from typing_extensions import TypedDict

pygram.initialize(cache_dir=None)


T = TypeVar("T")


def parse(code: str, grammar: Grammar = pygram.python_grammar_soft_keywords) -> NL:
    """Parse the given code using the given grammar."""
    driver = pygram.driver.Driver(grammar)
    return driver.parse_string(code, debug=True)


class UnsupportedSyntaxError(Exception):
    """Raised when some syntax is not supported on the current Python version."""


@dataclass
class Visitor(Generic[T]):
    token_type_to_name: Dict[int, str] = field(default_factory=lambda: token.tok_name)
    node_type_to_name: Callable[[int], str] = field(default_factory=lambda: type_repr)

    def get_node_name(self, node: NL) -> str:
        if node.type < 256:
            return self.token_type_to_name[node.type]
        else:
            return self.node_type_to_name(node.type)

    def visit(self, node: NL) -> T:
        name = self.get_node_name(node)
        method = getattr(self, f"visit_{name}", self.generic_visit)
        return method(node)

    def generic_visit(self, node: Node) -> T:
        raise NotImplementedError(f"visit_{self.get_node_name(node)}")


class LineRange(TypedDict):
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int


def get_line_range_for_leaf(leaf: Leaf) -> LineRange:
    num_newlines = leaf.value.count("\n")
    if num_newlines == 0:
        end_col_offset = leaf.column + len(leaf.value)
    else:
        end_col_offset = len(leaf.value) - leaf.value.rfind("\n") - 1
    return LineRange(
        lineno=leaf.lineno,
        col_offset=leaf.column,
        end_lineno=leaf.lineno + num_newlines,
        end_col_offset=end_col_offset,
    )


def get_line_range(node: NL) -> LineRange:
    if isinstance(node, Leaf):
        return get_line_range_for_leaf(node)
    else:
        begin_range = get_line_range(node.children[0])
        end_range = get_line_range(node.children[-1])
        return unify_line_ranges(begin_range, end_range)


def unify_line_ranges(begin_range: LineRange, end_range: LineRange) -> LineRange:
    return LineRange(
        lineno=begin_range["lineno"],
        col_offset=begin_range["col_offset"],
        end_lineno=end_range["end_lineno"],
        end_col_offset=end_range["end_col_offset"],
    )


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
    token.EQEQUAL: ast.Eq,
    token.NOTEQUAL: ast.NotEq,
    token.LESS: ast.Lt,
    token.GREATER: ast.Gt,
    token.LESSEQUAL: ast.LtE,
    token.GREATEREQUAL: ast.GtE,
}
TOKEN_TYPE_TO_UNARY_OP = {
    token.PLUS: ast.UAdd,
    token.MINUS: ast.USub,
    token.TILDE: ast.Invert,
}


class Compiler(Visitor[ast.AST]):
    def visit_typevar(self, node: Node) -> ast.AST:
        if sys.version_info >= (3, 12):
            bound = None
            default = None
            for index in (1, 3):
                if len(node.children) > index:
                    punc = node.children[index]
                    if punc.type == token.COLON:
                        bound = self.visit(node.children[index + 1])
                    elif punc.type == token.EQUAL:
                        default = self.visit(node.children[index + 1])
            if sys.version_info >= (3, 13):
                return ast.TypeVar(
                    name=node.children[0].value,
                    bound=bound,
                    default_value=default,
                    **get_line_range(node),
                )
            else:
                if default is not None and sys.version_info < (3, 13):
                    raise UnsupportedSyntaxError("TypeVar default")
                return ast.TypeVar(
                    name=node.children[0].value, bound=bound, **get_line_range(node)
                )
        else:
            raise UnsupportedSyntaxError("TypeVar")

    def visit_paramspec(self, node: Node) -> ast.AST:
        if sys.version_info >= (3, 13):
            if len(node.children) == 4:
                default_value = self.visit(node.children[3])
            else:
                default_value = None
            return ast.ParamSpec(
                name=node.children[0].value,
                default_value=default_value,
                **get_line_range(node),
            )
        if sys.version_info >= (3, 12):
            return ast.ParamSpec(name=node.children[0].value, **get_line_range(node))
        else:
            raise UnsupportedSyntaxError("ParamSpec")

    def visit_typevartuple(self, node: Node) -> ast.AST:
        if sys.version_info >= (3, 13):
            if len(node.children) == 4:
                default_value = self.visit(node.children[3])
            else:
                default_value = None
            return ast.TypeVarTuple(
                name=node.children[0].value,
                default_value=default_value,
                **get_line_range(node),
            )
        if sys.version_info >= (3, 12):
            return ast.TypeVarTuple(name=node.children[0].value, **get_line_range(node))
        else:
            raise UnsupportedSyntaxError("TypeVarTuple")

    def visit_typeparam(self, node: Node) -> ast.AST:
        return self.visit(node.children[0])

    def visit_file_input(self, node: Node) -> ast.AST:
        return ast.Module(
            # skip ENDMARKER
            body=[self.visit(child) for child in node.children[:-1]],
            type_ignores=[],  # TODO
            **get_line_range(node),
        )

    # Statements
    def visit_simple_stmt(self, node: Node) -> ast.AST:
        val = self.visit(node.children[0])
        if isinstance(val, ast.expr):
            return ast.Expr(val, **get_line_range(node.children[0]))
        else:
            return val

    # Expressions
    def visit_expr(self, node: Node) -> ast.AST:
        op = self.visit(node.children[0])
        begin_range = get_line_range(node.children[0])
        for child_index in range(2, len(node.children), 2):
            child = node.children[child_index]
            operator = node.children[child_index - 1]
            op = ast.BinOp(
                left=op,
                op=TOKEN_TYPE_TO_BINOP[operator.type](),
                right=self.visit(child),
                **unify_line_ranges(begin_range, get_line_range(child)),
            )
        return op

    def visit_factor(self, node: Node) -> ast.AST:
        return ast.UnaryOp(
            op=TOKEN_TYPE_TO_UNARY_OP[node.children[0].type](),
            operand=self.visit(node.children[1]),
            **get_line_range(node),
        )

    visit_xor_expr = visit_and_expr = visit_shift_expr = visit_arith_expr = (
        visit_term
    ) = visit_expr

    # Leaves
    def visit_NAME(self, leaf: Leaf) -> ast.AST:
        return ast.Name(id=leaf.value, ctx=ast.Load(), **get_line_range(leaf))

    def visit_NUMBER(self, leaf: Leaf) -> ast.AST:
        return ast.Constant(value=ast.literal_eval(leaf.value), **get_line_range(leaf))

    def visit_STRING(self, leaf: Leaf) -> ast.AST:
        return ast.Constant(value=ast.literal_eval(leaf.value), **get_line_range(leaf))


def compile(code: str) -> ast.AST:
    return Compiler().visit(parse(code + "\n"))


if __name__ == "__main__":
    import sys

    code = sys.argv[1]
    print(ast.dump(compile(code)))
