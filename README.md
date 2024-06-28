# lib2toast

This library converts a lib2to3 concrete syntax tree (CST) to a
standard Python AST.

## Python version support

This library supports Python versions 3.9 and up.

Python 3.8 is unsupported because it is about to reach the end of its
support period, the AST structure is quite different between 3.8 and 3.9,
and I don't have a use case for 3.8.

In the future I plan to support all supported upstream versions of Python.
