"""Exercise 08: topological sorting with Kahn's algorithm.

TODO:
- Input: number of nodes n, directed edges u->v.
- Return a valid topological ordering as a list of length n.
- Return [] if a cycle exists.
"""


def toposort(n: int, edges: list[tuple[int, int]]) -> list[int]:
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    order = toposort(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    assert order[0] == 0 and order[-1] == 3 and len(order) == 4
    assert toposort(2, [(0, 1), (1, 0)]) == []
    print("ok")
