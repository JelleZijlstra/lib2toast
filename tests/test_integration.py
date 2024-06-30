from collections.abc import Iterable
from pathlib import Path

import lib2toast
import pytest

from .checker import assert_compiles


def generate_test_cases() -> Iterable[Path]:
    test_dir = Path(__file__).parent
    for file_path in test_dir.glob("*.py"):
        yield file_path
    lib2toast_dir = Path(lib2toast.__file__).parent
    for file_path in lib2toast_dir.glob("*.py"):
        yield file_path


@pytest.mark.parametrize("file_path", generate_test_cases())
def test_python_file(file_path: Path) -> None:
    code = file_path.read_text(encoding="utf-8")
    assert_compiles(code)
