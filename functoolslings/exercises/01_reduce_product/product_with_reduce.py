"""Exercise 01: Use reduce to compute a product."""
from functools import reduce


def product_with_reduce(values: list[int], starting_integer: int = 1) -> int:
    # TODO: multiply instead of returning accumulator unchanged
    return reduce(lambda acc, x: acc * x, values, starting_integer)

if __name__ == "__main__":
    print(product_with_reduce([1,2,3]))
    print(product_with_reduce([10,11,12]))
