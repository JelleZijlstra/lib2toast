# lib2toast

This library converts a lib2to3 concrete syntax tree (CST) to a
standard Python AST.

Potential use cases include:

- Parsing Python code with less dependence on the Python version
- Extending or modifying the Python grammar in order to experiment
  with new features or to create a Python-like dialect

## Usage

This library is still at an early stage and the API may change.

- `lib2toast.api.compile(code, *, grammar=..., compiler=...)`: Compile
  a string of code to an AST. This AST can then be compiled to a Python
  code object or executed with the built-in `compile()` and `exec()` functions.
  By default, this uses a grammar that covers all syntax that is accepted
  by the latest version of Python, plus some additional syntax. Pass a custom
  _grammar_ to use different syntax. You can use `lib2toast.api.load_grammar` to
  load a grammar object from a file. If you do this, you'll usually also want
  to pass a custom `compiler` object by subclassing `lib2toast.compile.Compiler`.
- `lib2toast.api.run(code, *, filename=..., grammar=..., compiler=...)`: Compiles
  code and then immediately executes it.
- `lib2toast.api.load_grammar(path, *, async_keywords=True)`: Load a grammar
  file from a path. If `async_keywords` is True, treats `async` as a keyword
  as in Python 3.7+.

There is also a command-line interface: `python -m lib2toast -c code` runs
`code` after parsing it using lib2toast.

## Showcase

The command-line interface shows that lib2toast supports parsing some new syntax
in older Python versions:

```
$ python3.9 -m lib2toast -c 'print(f"{"x"}")'
x
```

This is new syntax introduced in Python 3.12 by PEP 701.

It also supports some (not all) Python 2 syntax that was removed in Python 3:

```
$ python3.9 -m lib2toast -c 'print(1 <> 2)'
True
```

The [test suite](./tests/test_custom_grammar.py) shows some examples of syntactic
variants of Python parsed with lib2toast. For example:

```
dataclass(frozen=True) C:
    x: int
    y: int = 0
```

## Implementation

lib2toast is implemented on top of `blib2to3`, the fork of `lib2to3` maintained
by the [Black](https://github.com/psf/black) project in order to parse and format
Python code. It originates from lib2to3, a tool shipped with earlier Python 3 versions
to support converting between Python 2 and 3 code.

The core part of the implementation is a tool that converts Python code to an
AST. This makes it easy to test for correctness: just run Python's built-in
`ast.parse` and assert that it produces the same tree, including line and
column numbers. So far I have tested the compiler on lib2toast's own code
as well as some of Black's code (the Black test cases were especially helpful),
but there are probably more bugs.

## Python version support

This library supports Python versions 3.9 and up.

Python 3.8 is unsupported because it is about to reach the end of its
support period, the AST structure is quite different between 3.8 and 3.9,
and I don't have a use case for 3.8.

In the future I plan to support all supported upstream versions of Python.

## Contributing

Contributions to this project are welcome, including ideas for new ways to
use the core functionality of the library.

Check the "Issues" tab for potential areas to contribute.
