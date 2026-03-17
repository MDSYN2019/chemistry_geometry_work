"""Exercise 03: Use partial to specialize a power function."""
from functools import partial


def power(base: int, exponent: int) -> int:
    return base ** exponent


def make_power_functions() -> tuple[callable, callable]:
    # TODO: create square and cube using partial(power, exponent=...)
    square = power
    cube = power
    return square, cube
