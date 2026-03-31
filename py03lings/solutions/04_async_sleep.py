"""Solution 04: write a simple async function and run it."""

import asyncio


async def delayed_upper(text: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return text.upper()


async def main() -> None:
    result = await delayed_upper("chemistry", 0.01)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
