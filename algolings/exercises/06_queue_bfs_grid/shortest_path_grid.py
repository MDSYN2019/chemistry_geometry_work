"""Exercise 06: BFS shortest path on a binary grid.

Grid cells: 0 = open, 1 = blocked.

TODO:
- Return shortest number of steps from start to goal using 4-neighbor moves.
- Return -1 if unreachable.
"""


def shortest_path_grid(grid: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> int:
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    grid = [[0, 0, 1], [1, 0, 1], [0, 0, 0]]
    assert shortest_path_grid(grid, (0, 0), (2, 2)) == 4
    assert shortest_path_grid(grid, (0, 0), (2, 0)) == -1
    print("ok")
