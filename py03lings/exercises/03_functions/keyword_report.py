"""Exercise 03: use defaults, *args, and **kwargs."""


def format_report(title: str, *items: str, uppercase: bool = False, **meta: str) -> str:
    # TODO: Build body from items joined by "; " (or "(none)" if empty)
    # TODO: Build metadata as "key=value" pairs sorted by key, joined by ", "
    # TODO: Final format: "[{title}] {body} | {metadata}"
    # TODO: If uppercase=True, uppercase the entire final string
    return ""


if __name__ == "__main__":
    print(format_report("Lab", "prep", "mix", owner="alex", room="B12"))
