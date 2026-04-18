"""Exercise 03: prefix sums.

TODO:
- Build a prefix-sum array.
- Implement range_sum(prefix, left, right) as inclusive range sum.
"""


def build_prefix(nums: list[int]) -> list[int]:
    # TODO
    raise NotImplementedError


def range_sum(prefix: list[int], left: int, right: int) -> int:
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    p = build_prefix([3, 1, 4, 1, 5])
    assert range_sum(p, 1, 3) == 6
    assert range_sum(p, 0, 4) == 14
    print("ok")
