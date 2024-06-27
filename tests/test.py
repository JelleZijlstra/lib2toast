import ast

from lib2toast.compile import compile


def assert_compiles(code: str) -> None:
    our_code = compile(code)
    ast_code = ast.parse(code)
    assert ast.dump(our_code, include_attributes=True) == ast.dump(
        ast_code, include_attributes=True
    )


def test_name() -> None:
    assert_compiles("a")


def test_literal() -> None:
    assert_compiles("1")
    assert_compiles("1.0")
    assert_compiles('"hello"')
    assert_compiles('"""hello"""')
    assert_compiles('r"hello"')


def test_arithmetic() -> None:
    assert_compiles("1 + 2")
    assert_compiles("1 + 2 + 3")
    assert_compiles("1 + 2 * 3")
    assert_compiles("1 | 3")
    assert_compiles("1 ^ 3")
    assert_compiles("1 & 3")
    assert_compiles("1 << 3")
    assert_compiles("1 >> 3")
    assert_compiles("1 // 3")
    assert_compiles("1 / 3")
    assert_compiles("1 % 3")
    assert_compiles("2 * 3 // 4")
    assert_compiles("1 + 2 * 3 // 4 % 5")
