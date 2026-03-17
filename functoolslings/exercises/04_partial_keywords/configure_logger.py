"""Exercise 04: Use partial keyword arguments."""
from functools import partial


def log(prefix: str, message: str, upper: bool = False) -> str:
    text = message.upper() if upper else message
    return f"[{prefix}] {text}"


def configure_logger(prefix: str):
    # TODO: return a logger with prefix fixed and upper=True
    return log
