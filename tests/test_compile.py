import sys

from .checker import assert_compiles


def test_name() -> None:
    assert_compiles("a")


def test_literal() -> None:
    assert_compiles("1")
    assert_compiles("1.0")
    assert_compiles('"hello"')
    assert_compiles('"""hello"""')
    assert_compiles('r"hello"')
    assert_compiles('"é"')
    assert_compiles('" "')  # non-breaking space
    assert_compiles('"✨ 🍰 ✨"')
    assert_compiles('u"x"')
    assert_compiles('U"x"')


def test_cake() -> None:
    assert_compiles('out(" ✨ 🍰 ✨")')


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
    assert_compiles("f(a[b])")


def test_named_expr() -> None:
    assert_compiles("(a := b)")


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
    if sys.version_info >= (3, 11):
        assert_compiles("a[*b]")
    if sys.version_info >= (3, 10):
        assert_compiles("a[b:=10]")
        assert_compiles("a[b:=10, c:=20]")


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
    assert_compiles('f"a"')
    assert_compiles('f"hello"')
    assert_compiles('f"{a}"')
    assert_compiles('f"{a}{b}"')
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
    assert_compiles('f"}}"')
    assert_compiles(r'rf"\{{"')
    assert_compiles("x = f'a'")
    assert_compiles('u"" f"x"')


def test_byte_string() -> None:
    assert_compiles("b''")
    assert_compiles("b'hello'")
    assert_compiles("b'hello' b'world'")
    assert_compiles("B'hello'")
    assert_compiles("B'hello' b'world'")
    assert_compiles(r"b'\x33'")
    assert_compiles(r"rb'\x33'")
    assert_compiles(r"b'\x33' b'\x44'")


def test_lambda() -> None:
    assert_compiles("lambda: 1")
    assert_compiles("lambda x: x")
    assert_compiles("lambda x, y: x + y")
    assert_compiles("lambda x=1: x")
    assert_compiles("lambda x=1, y=2: x + y")
    assert_compiles("lambda x, y=2, /: x + y")
    assert_compiles("lambda x, y=2, *, z=3: x + y + z")
    assert_compiles("lambda x, y=2, /, z=3: x + y + z")
    assert_compiles("lambda x, y=2, /, *, z=3: x + y + z")
    assert_compiles("lambda *args: 1")
    assert_compiles("lambda **kwargs: 1")
    assert_compiles("lambda x, *args: x")
    assert_compiles("lambda x, **kwargs: x")
    assert_compiles("lambda *args, x: x")


def test_assignment() -> None:
    assert_compiles("a = b")
    assert_compiles("a = b = c")
    assert_compiles("a, b = c")
    assert_compiles("a, b = c, d")
    assert_compiles("a.b = c")
    assert_compiles("a[b] = c[d] = e")
    assert_compiles("self.a[b] = c")
    assert_compiles("a[b].c = d")
    if sys.version_info >= (3, 11):
        assert_compiles("a[*b] = c")

    assert_compiles("a: int")
    assert_compiles("a: int = b")
    assert_compiles("(a): int = b")
    assert_compiles("a.b: int = c")
    assert_compiles("a[b]: int = e")
    assert_compiles("a += b")
    assert_compiles("a -= b")
    assert_compiles("a *= b")
    assert_compiles("a /= b")
    assert_compiles("a //= b")
    assert_compiles("a %= b")
    assert_compiles("a **= b")
    assert_compiles("a <<= b")
    assert_compiles("a >>= b")
    assert_compiles("a &= b")
    assert_compiles("a ^= b")
    assert_compiles("a |= b")
    assert_compiles("a.b += c")


def test_del() -> None:
    assert_compiles("del a")
    assert_compiles("del a.b")
    assert_compiles("del a[b]")
    assert_compiles("del a, b")
    assert_compiles("del (a, b)")


def test_return() -> None:
    assert_compiles("return 1")
    assert_compiles("return")


def test_simple_statements() -> None:
    assert_compiles("pass")
    assert_compiles("break")
    assert_compiles("continue")
    assert_compiles("a; b")


def test_raise() -> None:
    assert_compiles("raise")
    assert_compiles("raise a")
    assert_compiles("raise a from b")


def test_global_nonlocal() -> None:
    assert_compiles("global a")
    assert_compiles("global a, b")
    assert_compiles("nonlocal a")
    assert_compiles("nonlocal a, b")


def test_yield() -> None:
    assert_compiles("yield")
    assert_compiles("yield 1")
    assert_compiles("yield 1, 2")
    assert_compiles("(yield)")
    assert_compiles("(yield 1)")

    assert_compiles("yield from a")
    assert_compiles("yield from (a, b)")
    assert_compiles("(yield from a)")


if sys.version_info >= (3, 12):

    def test_type() -> None:
        assert_compiles("type x = int")
        assert_compiles("type x[T] = int")
        assert_compiles("type x[T,] = int")
        assert_compiles("type x[T] = int | str")
        assert_compiles("type x[*Ts] = int")
        assert_compiles("type x[T, U] = int")
        assert_compiles("type x[**P] = int")
        assert_compiles("type x[T: str] = int")
        assert_compiles("type x[T: str | int] = int")
        assert_compiles("type x[T: (str, int), U: float] = int")


if sys.version_info >= (3, 13):

    def test_type_param_default() -> None:
        assert_compiles("type x[T = int] = int")
        assert_compiles("type x[T = int, U = str] = int")
        assert_compiles("type x[T: str = int] = int")
        assert_compiles("type x[T: str | int = int] = int")
        assert_compiles("type x[T: (str, int) = int] = int")
        assert_compiles("type x[T: (str, int) = int, U: float = int] = int")
        assert_compiles("type x[*Ts = int] = int")
        assert_compiles("type x[*Ts = *int] = int")
        assert_compiles("type x[**P = int] = int")


def test_import() -> None:
    assert_compiles("import a")
    assert_compiles("import a as b")
    assert_compiles("import a.b")
    assert_compiles("import a.b as c")
    assert_compiles("import a, b")
    assert_compiles("import a . b")


def test_import_from() -> None:
    assert_compiles("from a import b")
    assert_compiles("from a import b as c")
    assert_compiles("from a import b, c")
    assert_compiles("from . import a")
    assert_compiles("from .. import a")
    assert_compiles("from .a import b")
    assert_compiles("from a import *")
    assert_compiles("from .a import *")
    assert_compiles("from a.b import c")


def test_if() -> None:
    assert_compiles("if a: pass")
    assert_compiles("if a: pass\nelse: pass")
    assert_compiles("if a: pass\nelif b: pass")
    assert_compiles("if a: pass\nelif b: pass\nelse: pass")
    assert_compiles("if a: 1")
    assert_compiles(
        """
        if a:
            pass
        """
    )
    assert_compiles(
        """
        if a:
            1
            pass
        elif b:
            2
            pass
        else:
            3
            pass
        """
    )


def test_while() -> None:
    assert_compiles("while a: pass")
    assert_compiles("while a: pass\nelse: pass")
    assert_compiles(
        """
        while a:
            pass
        """
    )
    assert_compiles(
        """
        while a:
            1
            pass
        else:
            2
            pass
        """
    )


def test_for() -> None:
    assert_compiles("for a in b: pass")
    assert_compiles("for a in b: pass\nelse: pass")
    assert_compiles(
        """
        for a in b:
            pass
        """
    )
    assert_compiles(
        """
        for a in b:
            1
            pass
        else:
            2
            pass
        """
    )


def test_with() -> None:
    assert_compiles("with a: pass")
    assert_compiles("with a as b: pass")
    assert_compiles("with a as b, c: pass")
    assert_compiles("with a as b, c as d: pass")
    assert_compiles("with a as b, c as d: 1")
    assert_compiles(
        """
        with a as b, c as d:
            1
            pass
        """
    )
    assert_compiles("with (a as b): pass")
    assert_compiles("with (a as b, c as d): pass")
    assert_compiles("with (a as b, c as d,): pass")


def test_async_for() -> None:
    assert_compiles("async for a in b: pass")
    assert_compiles("async for a in b: pass\nelse: pass")
    assert_compiles(
        """
        async for a in b:
            pass
        """
    )
    assert_compiles(
        """
        async for a in b:
            1
            pass
        else:
            2
            pass
        """
    )
    assert_compiles("with (a, b): pass")
    assert_compiles("with (a := b): pass")
    assert_compiles("with (a := b, c := d): pass")
    assert_compiles("with ((a := b), c := d): pass")
    assert_compiles("with ((a := b), (c := d)): pass")
    assert_compiles("with (a := b, c): pass")


def test_async_with() -> None:
    assert_compiles("async with a: pass")
    assert_compiles("async with a as b: pass")
    assert_compiles("async with a as b, c: pass")
    assert_compiles("async with a as b, c as d: pass")
    assert_compiles("async with a as b, c as d: 1")
    assert_compiles(
        """
        async with a as b, c as d:
            1
            pass
        """
    )


def test_try() -> None:
    assert_compiles("try: pass\nexcept: pass")
    assert_compiles("try: pass\nfinally: pass")
    assert_compiles("try: pass\nexcept a: pass")
    assert_compiles("try: pass\nexcept a as b: pass")
    assert_compiles("try: pass\nexcept a as b: pass\nelse: pass")
    assert_compiles("try: pass\nexcept a as b: pass\nelse: pass\nfinally: pass")
    assert_compiles(
        """
        try:
            pass
        except a as b:
            pass
        else:
            pass
        finally:
            pass
        """
    )


if sys.version_info >= (3, 11):

    def test_try_star() -> None:
        assert_compiles("try: pass\nexcept* a as b: pass")
        assert_compiles("try: pass\nexcept* b: pass")
        assert_compiles("try: pass\nexcept* a as c: pass")
        assert_compiles("try: pass\nexcept* a as b: pass\nelse: pass\nfinally: pass")
        assert_compiles(
            """
            try:
                pass
            except* b as c:
                pass
            else:
                pass
            finally:
                pass
            """
        )


def test_function_def() -> None:
    assert_compiles("def f(): pass")
    assert_compiles("def f(a): pass")
    assert_compiles("def f(a, b): pass")
    assert_compiles("def f(a=1): pass")
    assert_compiles("def f() -> int: pass")

    assert_compiles("async def f(): pass")
    assert_compiles("async def f(a): pass")
    assert_compiles("async def f() -> str: pass")

    assert_compiles("def f(x: int): pass")
    assert_compiles("def f(x: int, y: str): pass")
    assert_compiles("def f(x: int = 1): pass")
    assert_compiles("def f(*, x: int) -> str: pass")
    assert_compiles("def f(*, x: int = 1) -> str: pass")
    assert_compiles("def f(*args: int) -> str: pass")
    if sys.version_info >= (3, 11):
        assert_compiles("def f(*args: *Ts) -> str: pass")
    assert_compiles("def f(**kwargs: int) -> str: pass")

    if sys.version_info >= (3, 12):
        assert_compiles("def f[T](): pass")
        assert_compiles("def f[T, U](): pass")
        assert_compiles("async def f[T: int](): pass")


def test_class_def() -> None:
    assert_compiles("class A: pass")
    assert_compiles("class A(): pass")
    assert_compiles("class A(B): pass")
    assert_compiles("class A(B, C): pass")
    assert_compiles("class A(*args): pass")
    assert_compiles("class A(B, C, x=3): pass")
    assert_compiles("class A(B, C, x=3, **kwargs): pass")

    if sys.version_info >= (3, 12):
        assert_compiles("class A[T]: pass")
        assert_compiles("class A[T, U]: pass")
        assert_compiles("class A[T: int]: pass")

    assert_compiles(
        """
        class A:
            def f():
                pass
        """
    )


def test_decorators() -> None:
    assert_compiles("@a\ndef f(): pass")
    assert_compiles("@a\n@b\ndef f(): pass")
    assert_compiles("@a\n@b(c)\ndef f(): pass")
    assert_compiles("@a\ndef f(): pass\n@b\ndef g(): pass")
    assert_compiles("@a\ndef f(): pass\n@b(c)\ndef g(): pass")
    assert_compiles("@a\nasync def f(): pass")
    assert_compiles("@a\n@b\nclass f(): pass")


def test_empty() -> None:
    assert_compiles("")


def test_constants() -> None:
    assert_compiles("True")
    assert_compiles("False")
    assert_compiles("None")
    assert_compiles("...")


def test_starred() -> None:
    assert_compiles("a, *b = c")
    assert_compiles("*a, b = c")
    assert_compiles("a = (*b,)")
    assert_compiles("a = (*b, *c)")
    assert_compiles("a = [*b]")
    assert_compiles("a = {x, *c}")


if sys.version_info >= (3, 10):

    def test_match() -> None:
        assert_compiles(
            """
            match a:
                case b:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case (b):
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case "x":
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case 1:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case "x" "y":
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case b if c:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case b as c:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case _:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case _ as b:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case b, c:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case b.c:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case b.c.d:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case b.c.d.e:
                    pass
            """
        )
        assert_compiles(
            """
            match a, b:
                case b | c:
                    pass
            """
        )
        assert_compiles(
            """
            match a,:
                case b | c:
                    pass
            """
        )
        assert_compiles(
            """
            match a, b,:
                case b | c:
                    pass
            """
        )

    def test_match_sequence() -> None:
        assert_compiles(
            """
            match a:
                case (b, c):
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case (b, c, *d):
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case (b, c, *_):
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case ():
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case []:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case [b]:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case [b, *c]:
                    pass
            """
        )

    def test_match_mapping() -> None:
        assert_compiles(
            """
            match a:
                case {"b": c}:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case {}:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case {"b": c, **d}:
                    pass
            """
        )

    def test_match_or() -> None:
        assert_compiles(
            """
            match a:
                case b | c:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case b | c | d:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case b | c | [a, b]:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case (1 as b) | (2 as c) | [a, b]:
                    pass
            """
        )

    def test_match_class() -> None:
        assert_compiles(
            """
            match a:
                case int():
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case int(x):
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case int(x, y):
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case int(x, y, z=3):
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case int(42, z=5):
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case int(a as b):
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case int(a=re.match(a)):
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case a.b.c(a=b):
                    pass
            """
        )

    def test_match_numerics() -> None:
        assert_compiles(
            """
            match a:
                case -1:
                    pass
            """
        )
        assert_compiles(
            """
            match a:
                case 1+2j:
                    pass
            """
        )
