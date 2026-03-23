"""Exercise 00: Use reduce to sum a list of integers."""
from functools import reduce


def sum_with_reduce(values: list[int]) -> int:
    # TODO: replace lambda body so the running total grows correctly
    return reduce(lambda acc, x: acc, values, 0)


def concatenate_list(values : list[str]) -> str:
    return reduce(lambda x, y: x + " " + y, values)


if __name__ == "__main__":
    print(sum_with_reduce([1, 2, 3, 4]))
    print(concatenate_list(["sang", "young", "noh"]))
