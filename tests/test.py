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


def test_unary() -> None:
    assert_compiles("-1")
    assert_compiles("+1")
    assert_compiles("~1")


def test_power() -> None:
    assert_compiles("1 ** 2")
    assert_compiles("1 ** 2 ** 3")
    assert_compiles("1 ** 2 ** 3 ** 4")

    assert_compiles("await x")
    assert_compiles("await x ** 2")

    assert_compiles("a.b")
    assert_compiles("a.b.c")
    assert_compiles("a[b]")
    assert_compiles("a[b].c")
    assert_compiles("await a[b].c ** 2")


def test_call() -> None:
    assert_compiles("f()")
    assert_compiles("f(1)")
    assert_compiles("f(1, 2)")
    assert_compiles("f(a=1)")
    assert_compiles("f(a=1, b=2)")
    assert_compiles("f(1, a=1)")
    assert_compiles("f(*a)")
    assert_compiles("f(**a)")
    assert_compiles("f(1, *a)")
    assert_compiles("f(1, **a)")
    assert_compiles("f(a := b)")
    assert_compiles("f(a := b, c := d)")
    assert_compiles("f(a for b in c)")
    assert_compiles("f(a := b for c in d)")
    assert_compiles("f(a for b in c for d in e)")
    assert_compiles("f(a for b in c if d)")
    assert_compiles("f(a for b in c if d for e in f)")
    assert_compiles("f(a for b in c if d for e in f if g)")
    assert_compiles("f(a async for b in c)")
    assert_compiles("f(a async for b in c for d in e)")
