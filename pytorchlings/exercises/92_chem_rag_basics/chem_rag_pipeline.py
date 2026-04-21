"""Exercise 92: retrieval-augmented generation (RAG) basics for chemistry.

Goal:
- build a tiny retrieval index over chemistry notes
- retrieve top-k passages for a user question
- construct a grounded prompt context for an LLM
"""

from __future__ import annotations

from collections import Counter
import math


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in text.replace("/", " ").replace("-", " ").split()]


def cosine_sim(a: Counter[str], b: Counter[str]) -> float:
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve_top_k(query: str, docs: list[str], k: int = 2) -> list[tuple[int, float]]:
    q = Counter(tokenize(query))
    scored: list[tuple[int, float]] = []
    for i, doc in enumerate(docs):
        d = Counter(tokenize(doc))
        # TODO: replace placeholder with cosine_sim(q, d).
        score = 0.0
        scored.append((i, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def build_grounded_prompt(question: str, docs: list[str], top_hits: list[tuple[int, float]]) -> str:
    context = "\n".join([f"[doc {i}] {docs[i]}" for i, _ in top_hits])
    # TODO: return a concise grounded prompt that includes context + question.
    return f"Question: {question}\nContext:\n{context}"


if __name__ == "__main__":
    docs = [
        "Ethanol has molecular formula C2H6O and is a polar solvent.",
        "Benzene is aromatic and has six pi electrons.",
        "Acetic acid is a weak acid commonly written as CH3COOH.",
    ]
    query = "Which molecule is aromatic?"

    hits = retrieve_top_k(query, docs, k=2)
    prompt = build_grounded_prompt(query, docs, hits)

    assert len(hits) == 2
    print("retrieved:", hits)
    print(prompt)
    print("exercise 92 scaffold ready")
