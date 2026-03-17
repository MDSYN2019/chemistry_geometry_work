"""Solution 03."""
from functools import partial


def power(base: int, exponent: int) -> int:
    return base ** exponent


def make_power_functions() -> tuple[callable, callable]:
    square = partial(power, exponent=2)
    cube = partial(power, exponent=3)
    return square, cube
