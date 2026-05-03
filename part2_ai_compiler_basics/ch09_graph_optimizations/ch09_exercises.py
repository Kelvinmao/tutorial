"""
Chapter 9 — Exercise stubs.

Fill in each function, then run:
    pytest -m exercise -k ch09 -v
"""

from __future__ import annotations


def dead_node_elimination(
    nodes: list[dict],
    output_names: set[str],
) -> list[dict]:
    """Exercise 9.3 — Dead node elimination on a computation graph.

    Each node is a dict::

        {"name": str, "op": str, "inputs": list[str], "output": str}

    Remove every node whose ``output`` is **not** transitively needed to
    produce any name in *output_names*.  Return the surviving nodes in
    their original order.

    Algorithm sketch:
        1. Start from *output_names* as the "live" set.
        2. Walk nodes in reverse order; if a node's output is live, mark
           all its inputs as live too.
        3. Keep only nodes whose output is in the live set.
    """
    raise NotImplementedError("Exercise 9.3: implement dead_node_elimination")
