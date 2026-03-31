"""Solution 05: run async tasks concurrently with gather."""

import asyncio


async def fetch_label(label: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"done:{label}"


async def run_batch() -> list[str]:
    return await asyncio.gather(
        fetch_label("A", 0.02),
        fetch_label("B", 0.01),
        fetch_label("C", 0.03),
    )


async def main() -> None:
    results = await run_batch()
    print(", ".join(results))


if __name__ == "__main__":
    asyncio.run(main())
