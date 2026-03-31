"""Exercise 05: run async tasks concurrently with gather."""

import asyncio


async def fetch_label(label: str, delay: float) -> str:
    # TODO: await asyncio.sleep(delay)
    # TODO: return f"done:{label}"
    return ""


async def run_batch() -> list[str]:
    # TODO: run fetch_label for A (0.02), B (0.01), C (0.03) concurrently
    # TODO: return results from asyncio.gather(...)
    return []


async def main() -> None:
    results = await run_batch()
    print(", ".join(results))


if __name__ == "__main__":
    asyncio.run(main())
