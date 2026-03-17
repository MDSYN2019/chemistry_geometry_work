"""Solution 06."""
from functools import wraps


def trace(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        print(f"calling {fn.__name__}")
        return fn(*args, **kwargs)

    return inner
