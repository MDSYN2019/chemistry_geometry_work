"""Exercise 07: Use update_wrapper on callable class wrapper."""
from functools import update_wrapper


class CallCounter:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0
        # TODO: call update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


def build_wrapper(fn):
    return CallCounter(fn)
