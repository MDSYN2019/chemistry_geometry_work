#!/usr/bin/env python3
"""Inventory Boltz result files and print scalar confidence/affinity fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


def scalar_fields(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield dotted paths for JSON scalar values, avoiding large numeric arrays."""
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from scalar_fields(child, path)
    elif isinstance(value, list):
        if len(value) <= 10 and all(not isinstance(item, (dict, list)) for item in value):
            yield prefix, value
    elif value is None or isinstance(value, (str, int, float, bool)):
        yield prefix, value


def inspect(root: Path) -> int:
    if not root.is_dir():
        print(f"error: result directory does not exist: {root}")
        return 2

    files = sorted(path for path in root.rglob("*") if path.is_file())
    print(f"Found {len(files)} file(s) below {root}")
    for path in files:
        print(f"- {path.relative_to(root)}")

    json_files = [path for path in files if path.suffix.lower() == ".json"]
    for path in json_files:
        print(f"\n[{path.relative_to(root)}]")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            print(f"  could not read JSON: {error}")
            continue
        fields = list(scalar_fields(data))
        if not fields:
            print("  (no compact scalar fields; inspect the file directly)")
        for name, value in fields:
            print(f"  {name}: {value}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    args = parser.parse_args()
    return inspect(args.result_directory)


if __name__ == "__main__":
    raise SystemExit(main())
