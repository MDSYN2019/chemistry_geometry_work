"""Tiny rustlings-like helper: list exercises and count TODOs."""
from pathlib import Path


def main() -> None:
    root = Path(__file__).parent / "exercises"
    files = sorted(root.glob("**/*.py"))
    for file in files:
        todo_count = file.read_text().count("TODO")
        status = "✅" if todo_count == 0 else f"🧩 {todo_count} TODOs"
        print(f"{file.relative_to(root)} -> {status}")


if __name__ == "__main__":
    main()
