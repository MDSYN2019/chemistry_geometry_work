"""Solution 11."""
from functools import lru_cache


@lru_cache(maxsize=16, typed=True)
def add_one(x):
    return x + 1
