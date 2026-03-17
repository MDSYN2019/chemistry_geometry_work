"""Solution 31."""
from functools import lru_cache, partial, singledispatch


@singledispatch
def normalize(value):
    return str(value)


@normalize.register
def _(value: str):
    return value.strip().lower()


@lru_cache(maxsize=64)
def score(text: str) -> int:
    return sum(ord(ch) for ch in text)


def run_demo(values: list[object]) -> list[int]:
    convert = partial(map, normalize)
    normalized = list(convert(values))
    return [score(str(v)) for v in normalized]
