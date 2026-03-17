"""Exercise 01: Use reduce to compute a product."""
from functools import reduce


def product_with_reduce(values: list[int]) -> int:
    # TODO: multiply instead of returning accumulator unchanged
    return reduce(lambda acc, x: acc, values, 1)
