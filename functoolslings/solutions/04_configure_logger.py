"""Solution 04."""
from functools import partial


def log(prefix: str, message: str, upper: bool = False) -> str:
    text = message.upper() if upper else message
    return f"[{prefix}] {text}"


def configure_logger(prefix: str):
    return partial(log, prefix, upper=True)
