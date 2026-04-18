"""Exercise 09: union-find (disjoint set union).

TODO:
- Complete DSU with path compression + union by rank/size.
- Support connected(a, b) queries.
"""


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        # TODO
        raise NotImplementedError

    def union(self, a: int, b: int) -> bool:
        # TODO
        raise NotImplementedError

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)


if __name__ == "__main__":
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.connected(0, 2) is True
    assert uf.connected(0, 3) is False
    print("ok")
