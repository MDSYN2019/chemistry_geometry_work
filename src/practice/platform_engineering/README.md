# Platform Engineering Practice Pack (C# + Python)

This folder contains focused practice code and exercises for:

1. Distributed systems implementation (C# and Python)
2. Kubernetes multi-tenant platform operations
3. Docker/Terraform/CI-CD in controlled environments
4. Networking, storage, security, and performance tradeoffs
5. Model serving and inference infrastructure
6. Communication and technical documentation

## Files

- `distributed_queue_sim.py`: Python simulation of a partitioned queue and consumer group with lag/rebalance behavior.
- `ResilientWorker.cs`: C# background worker pattern with retries, jitter, timeout, and dead-letter handling.
- `k8s_multitenant_practice.md`: Kubernetes and platform extension exercises.
- `iac_and_cicd_practice.md`: Docker/Terraform/CI/CD + compliance-oriented tasks.
- `model_serving_practice.md`: inference architecture and scaling drills.

## Suggested progression

1. Run `distributed_queue_sim.py` and tune partition count / consumer count.
2. Port or extend retry logic between C# and Python implementations.
3. Complete one multi-tenant Kubernetes exercise and write an ADR.
4. Draft a regulated CI/CD control matrix for one service.
5. Build a small model-serving benchmark script and document latency tradeoffs.
