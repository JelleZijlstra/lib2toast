"""Debugging helpers."""

import sys

from .compile import parse

if __name__ == "__main__":
    code = sys.argv[1]
    print(repr(parse(code + "\n")))
