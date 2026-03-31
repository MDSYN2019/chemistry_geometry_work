"""Exercise 04: write a simple async function and run it."""

import asyncio


async def delayed_upper(text: str, delay: float) -> str:
    # TODO: await asyncio.sleep(delay)
    # TODO: return the uppercased text
    return ""


async def main() -> None:
    result = await delayed_upper("chemistry", 0.01)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
