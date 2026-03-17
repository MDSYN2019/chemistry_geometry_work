"""Exercise 05: Compose partial callables for text normalization."""
from functools import partial


def normalize(text: str, strip: bool = True, lower: bool = True) -> str:
    if strip:
        text = text.strip()
    if lower:
        text = text.lower()
    return text


def run_pipeline(texts: list[str]) -> list[str]:
    cleaner = partial(normalize, strip=True, lower=True)
    # TODO: apply cleaner to each text
    return texts
