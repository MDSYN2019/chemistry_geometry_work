"""Solution 02."""
from functools import reduce


def max_with_reduce(values: list[int]) -> int:
    return reduce(lambda acc, x: acc if acc > x else x, values)
