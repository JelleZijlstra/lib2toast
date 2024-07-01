import argparse
from pathlib import Path

from lib2toast.api import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run Python code parsed with lib2toast")
    parser.add_argument("path", type=Path, nargs="?")
    parser.add_argument("-c", "--code", type=str, default=None)
    args = parser.parse_args()
    if args.code is not None:
        if args.path is not None:
            parser.error("cannot specify both code and path")
        run(args.code)
    else:
        if args.path is None:
            parser.error("must specify either code or path")
        path = args.path
        if path.is_file():
            run(path.read_text(encoding="utf-8"))
        else:
            parser.error("path must be a file")
