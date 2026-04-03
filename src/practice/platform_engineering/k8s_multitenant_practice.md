# Kubernetes Multi-Tenant Practice Exercises

## Goal
Practice running a secure multi-tenant platform while keeping operational overhead manageable.

## Exercise 1: Tenant isolation baseline
- Create per-tenant namespaces (`tenant-a`, `tenant-b`).
- Add `ResourceQuota` and `LimitRange`.
- Add `NetworkPolicy` default deny + allow only ingress via shared gateway namespace.
- Write down which controls provide hard isolation vs soft isolation.

## Exercise 2: Admission control and policy
- Define policies (OPA Gatekeeper or Kyverno) to enforce:
  - required labels (`owner`, `cost-center`, `data-classification`)
  - disallow privileged containers
  - disallow `latest` tags
- Add an exemption process for break-glass workloads.
- Document how exemptions are audited and expired.

## Exercise 3: Platform extension
- Implement a basic CRD for a tenant-scoped service (e.g., `InferenceServiceProfile`).
- Add a controller/reconciler design note:
  - desired spec fields
  - status conditions
  - retry/backoff strategy
  - idempotency expectations

## Exercise 4: Multi-tenant observability
- Define logging dimensions: tenant, environment, service, request-id.
- Create SLOs with tenant-level burn-rate alerts.
- Prevent cross-tenant metric leakage in dashboards.

## Exercise 5: Failure scenario
- Simulate noisy neighbor CPU/memory pressure.
- Record effects on p95 latency for unaffected tenants.
- Propose mitigation: priority classes, bin packing, and HPA/VPA settings.
