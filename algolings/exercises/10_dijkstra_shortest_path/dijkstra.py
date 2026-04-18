"""Exercise 10: Dijkstra shortest path.

TODO:
- Graph is adjacency list: graph[u] = [(v, weight), ...].
- Return distance from source to target, or -1 if unreachable.
"""


def dijkstra(graph: list[list[tuple[int, int]]], source: int, target: int) -> int:
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    g = [
        [(1, 4), (2, 1)],
        [(3, 1)],
        [(1, 2), (3, 5)],
        [],
    ]
    assert dijkstra(g, 0, 3) == 4
    assert dijkstra(g, 3, 0) == -1
    print("ok")
