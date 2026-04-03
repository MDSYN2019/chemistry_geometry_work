# Model Serving and Inference Infrastructure Practice

## Exercise 1: Serving architecture
Design two paths for open-model inference:
- low-latency online API
- high-throughput async batch

For each path define:
- autoscaling signal (QPS, queue depth, token/s)
- concurrency and batching policy
- timeout and retry strategy
- fallback/degradation behavior

## Exercise 2: Performance profiling
- Measure p50/p95 latency under increasing concurrent load.
- Track token throughput and GPU/CPU utilization.
- Identify saturation point and bottleneck type:
  - compute-bound
  - memory-bound
  - network-bound

## Exercise 3: Cost/performance tuning
- Compare quantization options and context window sizes.
- Evaluate dynamic batching vs tail latency impact.
- Define safe default limits per tenant.

## Exercise 4: Reliability and safety
- Add request validation, prompt size guards, and rate limits.
- Design DLQ flow for async inference failures.
- Specify model version rollback strategy.

## Exercise 5: Technical communication
Write:
- an ADR for chosen serving architecture
- a runbook for on-call response to elevated latency
- a dashboard legend explaining each critical metric in plain language
