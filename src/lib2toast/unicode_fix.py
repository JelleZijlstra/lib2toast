"""Fix column offsets when Unicode is involved."""

from collections.abc import Iterable

from blib2to3.pytree import NL, Leaf


def dfs(node: NL) -> Iterable[Leaf]:
    if isinstance(node, Leaf):
        yield node
        return
    for child in node.children:
        if isinstance(child, Leaf):
            yield child
        else:
            yield from dfs(child)


def fixup_unicode(node: NL) -> None:
    current_lineno = 0
    current_coloffset = 0
    for leaf in dfs(node):
        if leaf.lineno != current_lineno:
            current_coloffset = 0
            current_lineno = leaf.lineno

        if leaf.prefix:
            if "\n" in leaf.prefix:
                prefix = leaf.prefix.rsplit("\n", maxsplit=1)[-1]
            else:
                prefix = leaf.prefix
            current_coloffset += len(prefix.encode("utf-8"))
        leaf.column = current_coloffset
        if "\n" in leaf.value:
            current_coloffset = len(
                leaf.value.rsplit("\n", maxsplit=1)[-1].encode("utf-8")
            )
            current_lineno += leaf.value.count("\n")
        else:
            current_coloffset += len(leaf.value.encode("utf-8"))
