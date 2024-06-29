import argparse
import ast
import difflib
import pathlib
import sys
import textwrap
import traceback
from pathlib import Path

from lib2toast.compile import compile


def assert_compiles(code: str) -> None:
    code = textwrap.dedent(code)
    our_code = compile(code)
    ast_code = ast.parse(code)
    if ast.dump(our_code, include_attributes=True, indent=2) != ast.dump(
        ast_code, include_attributes=True, indent=2
    ):
        diff = "\n".join(
            difflib.unified_diff(
                ast.dump(our_code, include_attributes=True, indent=2).splitlines(),
                ast.dump(ast_code, include_attributes=True, indent=2).splitlines(),
                "lib2toast",
                "ast",
            )
        )
        raise AssertionError(f"Code is not equal\n{diff}")


def check_file(path: Path) -> bool:
    text = path.read_text()
    try:
        our_code = compile(text)
    except Exception:
        print(f"{'='*80}\n{path}\n{'='*80}")
        traceback.print_exc()
        return False
    ast_code = ast.parse(text)

    our_dump = ast.dump(our_code, include_attributes=True, indent=2)
    ast_dump = ast.dump(ast_code, include_attributes=True, indent=2)

    if our_dump != ast_dump:
        print(f"{'='*80}\n{path}\n{'='*80}")
        for line in difflib.unified_diff(
            our_dump.splitlines(), ast_dump.splitlines(), "lib2toast", "ast"
        ):
            print(line)
        return False
    return True


def check_directory(directory: Path) -> bool:
    result = True
    count = 0
    for path in directory.rglob("*.py"):
        count += 1
        if not check_file(path):
            result = False
    print(f"Checked {count} files")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=pathlib.Path)
    args = parser.parse_args()
    path = args.path
    if path.is_file():
        result = check_file(path)
    else:
        result = check_directory(path)
    sys.exit(0 if result else 1)
