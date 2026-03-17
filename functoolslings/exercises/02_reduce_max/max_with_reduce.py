"""Exercise 02: Use reduce to compute max value."""
from functools import reduce


def max_with_reduce(values: list[int]) -> int:
    # TODO: compare acc and x and keep larger value
    return reduce(lambda acc, x: acc, values)
