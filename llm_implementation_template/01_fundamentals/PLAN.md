# Stage 1: LLM App Fundamentals

## Deliverable
A minimal RAG + tool-use service that returns validated structured outputs.

## Components
1. Prompt templates (`system`, `task`, `few-shot`)
2. Ingestion + chunking pipeline
3. Embedding generation + vector index
4. Retrieval layer (top-k + filters)
5. Tool registry and function schemas
6. Structured output schema + validator

## Exit criteria
- Retrieval precision at k measured on a small golden set.
- Output schema validation pass rate >= target.
- Tool calls logged with argument validation.
