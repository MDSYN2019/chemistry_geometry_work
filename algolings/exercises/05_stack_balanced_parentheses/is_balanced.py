"""Exercise 05: stack-based delimiter checking.

TODO:
- Return True when (), [], {} are balanced.
- Return False otherwise.
"""


def is_balanced(text: str) -> bool:
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    assert is_balanced("{[()]}") is True
    assert is_balanced("([)]") is False
    print("ok")
