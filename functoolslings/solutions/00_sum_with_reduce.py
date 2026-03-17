"""Solution 00."""
from functools import reduce


def sum_with_reduce(values: list[int]) -> int:
    return reduce(lambda acc, x: acc + x, values, 0)
