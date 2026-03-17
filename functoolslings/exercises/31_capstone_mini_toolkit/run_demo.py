"""Exercise 31: Capstone toolkit.
Combine functools.partial, functools.singledispatch, and functools.lru_cache.
"""
from functools import lru_cache, partial, singledispatch


@singledispatch
def normalize(value):
    # TODO: provide default string conversion
    return value


@normalize.register
def _(value: str):
    return value.strip().lower()


@lru_cache(maxsize=64)
def score(text: str) -> int:
    return sum(ord(ch) for ch in text)


def run_demo(values: list[object]) -> list[int]:
    convert = partial(map, normalize)
    normalized = list(convert(values))
    # TODO: score normalized values as strings
    return []
