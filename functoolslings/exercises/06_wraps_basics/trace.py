"""Exercise 06: Use wraps to preserve metadata."""
from functools import wraps


def trace(fn):
    # TODO: add @wraps(fn) to the inner function
    def inner(*args, **kwargs):
        print(f"calling {fn.__name__}")
        return fn(*args, **kwargs)

    return inner
