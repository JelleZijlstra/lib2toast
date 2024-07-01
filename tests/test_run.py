import os
import textwrap
from typing import Any

from lib2toast.api import run


def check_run(code: str, expected_vars: dict[str, Any]) -> None:
    ns = run(textwrap.dedent(code))
    for key, value in expected_vars.items():
        assert ns[key] == value


def test_run_basic() -> None:
    check_run("a = 1", {"a": 1})
    check_run("import os", {"os": os})


def test_run_py2() -> None:
    check_run("x = 1 <> 2", {"x": True})
    check_run("x = 1 <> 1", {"x": False})


def test_repr() -> None:
    check_run("x = `'x'`", {"x": "'x'"})
    check_run("repr = 42; x = `'x'`", {"x": "'x'"})


def test_py2_except() -> None:
    code = """
        try:
            raise ValueError
        except ValueError, e:
            x = type(e)
    """
    check_run(code, {"x": ValueError})


def test_pep_701() -> None:
    check_run('x = f"a{f"b{1}c"}d"', {"x": "ab1cd"})
