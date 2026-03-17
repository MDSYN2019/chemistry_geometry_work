"""Exercise 11: Use typed lru_cache behavior."""
from functools import lru_cache


@lru_cache(maxsize=16, typed=False)
def add_one(x):
    # TODO: set typed=True so int and float cache separately
    return x + 1
