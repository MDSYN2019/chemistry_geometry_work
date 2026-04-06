# LLM Implementation Template

This folder is a practical template for implementing the sequence you outlined:

1. **LLM app fundamentals**
2. **Evaluation**
3. **Serving and ops**
4. **Applied ML depth**
5. **Optional specialization**

Each stage has a `PLAN.md` (what to build), a `CHECKLIST.md` (how to know it's done), and starter placeholders where useful.

---

## Structure

- `01_fundamentals/`
  - Build your first reliable retrieval + tool-using app.
- `02_evaluation/`
  - Add quality gates with golden sets and regression testing.
- `03_serving_ops/`
  - Move from notebook/demo to production operations.
- `04_applied_ml_depth/`
  - Add deeper ML improvements (reranking, finetuning, baselines).
- `05_optional_specialization/`
  - Choose one specialization track.
- `shared/`
  - Reusable configs, schemas, and metrics definitions.

---

## How each component works

### 1) Fundamentals
**Goal:** turn a model call into a dependable application.

- **Prompt design:** controls behavior and output style.
- **Embeddings:** convert text to vectors for semantic search.
- **Chunking:** splits documents into retrievable pieces.
- **Retrieval:** finds relevant chunks for each query.
- **Structured outputs:** forces machine-readable responses (e.g., JSON schema).
- **Tool use:** enables function/API calls for external actions or data.

### 2) Evaluation
**Goal:** measure quality continuously instead of manually spot-checking.

- **Golden set:** trusted examples with expected answers/behavior.
- **Regression tests:** ensure changes do not break known-good behavior.
- **Pairwise comparison:** compare candidate systems A vs. B.
- **Failure taxonomy:** classify errors (retrieval miss, hallucination, format failure, etc.).
- **Business metrics:** connect model quality to outcomes (conversion, time saved, cost).

### 3) Serving and Ops
**Goal:** operate the system safely at scale.

- **APIs:** stable entrypoints for product integration.
- **Async jobs/queues:** handle long-running generation and retries.
- **Caching:** reduce latency and cost for repeated requests.
- **Observability:** logs, traces, and dashboards for debugging quality and performance.
- **Cost/latency controls:** track token spend and optimize p50/p95 latency.

### 4) Applied ML Depth
**Goal:** improve ranking/quality beyond baseline RAG.

- **PyTorch basics:** ability to train/evaluate simple models.
- **Fine-tuning concepts:** adapt models with task-specific data.
- **Rerankers:** improve top-k retrieval ordering.
- **Embeddings iteration:** compare embedding models and chunk strategies.
- **Classical baselines:** sanity check with non-LLM methods.

### 5) Optional Specialization
**Goal:** choose a domain where advanced depth matters.

Pick one track and define:
- target users,
- key workflows,
- specialized evaluation criteria,
- unique risk controls.

---

## Suggested workflow

1. Complete `01_fundamentals` and ship a narrow MVP.
2. Build `02_evaluation` before broad rollout.
3. Add `03_serving_ops` before production traffic.
4. Use `04_applied_ml_depth` for measurable quality lifts.
5. Expand with one `05_optional_specialization` track.
