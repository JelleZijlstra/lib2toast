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
    assert_compiles("not 1")


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


def test_compare() -> None:
    assert_compiles("1 < 2")
    assert_compiles("1 <= 2")
    assert_compiles("1 == 2")
    assert_compiles("1 != 2")
    assert_compiles("1 > 2")
    assert_compiles("1 >= 2")
    assert_compiles("1 in 2")
    assert_compiles("1 not in 2")
    assert_compiles("1 is 2")
    assert_compiles("1 is not 2")
    assert_compiles("1 < 2 < 3")
    assert_compiles("1 < 2 <= 3")


def test_boolop() -> None:
    assert_compiles("1 and 2")
    assert_compiles("1 or 2")
    assert_compiles("1 and 2 or 3")
    assert_compiles("1 or 2 and 3")
    assert_compiles("1 and 2 and 3 and 4")
    assert_compiles("1 or 2 or 3 or 4")


def test_subscript() -> None:
    assert_compiles("a[1]")
    assert_compiles("a[1, 2]")
    assert_compiles("a[1:2]")
    assert_compiles("a[1:2, 3:4]")
    assert_compiles("a[1:2:3]")
    assert_compiles("a[1:2:3, 4:5:6]")
    assert_compiles("a[1, 2:3]")
    assert_compiles("a[:]")
    assert_compiles("a[::]")
    assert_compiles("a[1:]")
    assert_compiles("a[:2]")
    assert_compiles("a[::2]")


def test_atom() -> None:
    assert_compiles("[]")
    assert_compiles("[1]")
    assert_compiles("[*1]")
    assert_compiles("[1, 2]")
    assert_compiles("[1 for a in b]")
    assert_compiles("[1 for a in b for c in d]")
    assert_compiles("[1 for a in b if c]")
    assert_compiles("{}")
    assert_compiles("{1}")
    assert_compiles("{1: 2}")
    assert_compiles("{1: 2, 3: 4}")
    assert_compiles("{1: 2 for a in b}")
    assert_compiles("{1: 2 for a in b for c in d}")
    assert_compiles("{1: 2 for a in b if c}")
    assert_compiles("{1, 2}")
    assert_compiles("{**a}")
    assert_compiles("{1: 2, **a}")
    assert_compiles("{*a}")
    assert_compiles("(1)")
    assert_compiles("(1,)")
    assert_compiles("(1, 2)")
    assert_compiles("(1, 2,)")
    assert_compiles("(1 for a in b)")


def test_fstring() -> None:
    assert_compiles('f""')
    assert_compiles('f"hello"')
    assert_compiles('f"{a}"')
    assert_compiles('"a" "b"')
    assert_compiles('f"a" "b"')
    assert_compiles('f"{a!s}"')
    assert_compiles('f"{a!r}"')
    assert_compiles('f"{a!a}"')
    assert_compiles('f"{a!s:a}"')
    assert_compiles('f"{a!s:}"')
    assert_compiles('f"{a:{a}}"')
    assert_compiles('f"{a=}"')
    assert_compiles('f"{a=:b}"')
    assert_compiles('f"{a=}" "b"')
    assert_compiles('"a" f"{a=}"')
    assert_compiles('f"a{a=}"')
    assert_compiles('f"{a=}b"')
    assert_compiles('f"{a = }"')
    assert_compiles('f"{a = :b}"')
    assert_compiles('f"{a = !s}"')
    assert_compiles('f"{a = !r}"')
    assert_compiles('f"{a = :b!s}"')
    assert_compiles('f"{{"')


def test_byte_string() -> None:
    assert_compiles("b''")
    assert_compiles("b'hello'")
    assert_compiles("b'hello' b'world'")
    assert_compiles("B'hello'")
    assert_compiles("B'hello' b'world'")
    assert_compiles(r"b'\x33'")
    assert_compiles(r"rb'\x33'")
    assert_compiles(r"b'\x33' b'\x44'")
