"""Solution 05."""
from functools import partial


def normalize(text: str, strip: bool = True, lower: bool = True) -> str:
    if strip:
        text = text.strip()
    if lower:
        text = text.lower()
    return text


def run_pipeline(texts: list[str]) -> list[str]:
    cleaner = partial(normalize, strip=True, lower=True)
    return [cleaner(text) for text in texts]
