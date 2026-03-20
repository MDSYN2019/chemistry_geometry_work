"""Solution 03: use defaults, *args, and **kwargs."""


def format_report(title: str, *items: str, uppercase: bool = False, **meta: str) -> str:
    body = "; ".join(items) if items else "(none)"
    metadata = ", ".join(f"{k}={v}" for k, v in sorted(meta.items()))
    output = f"[{title}] {body} | {metadata}"
    return output.upper() if uppercase else output


if __name__ == "__main__":
    print(format_report("Lab", "prep", "mix", owner="alex", room="B12"))
