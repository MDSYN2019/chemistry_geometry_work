"""Exercise 07: merge intervals.

TODO:
- Given [start, end] intervals, merge overlaps.
- Return merged intervals sorted by start.
"""


def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]
    assert merge_intervals([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    print("ok")
