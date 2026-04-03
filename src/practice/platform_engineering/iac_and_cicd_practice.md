# Docker / Terraform / CI-CD Practice (Regulated Environment)

## Exercise 1: Reproducible image builds
- Build a minimal runtime image (distroless or slim base).
- Pin all package versions.
- Generate SBOM and sign image artifacts.
- Enforce image scan gate in CI.

## Exercise 2: Terraform controls
- Split environments with separate state backends and least-privilege IAM.
- Add policy checks (e.g., sentinel/OPA/tflint/checkov equivalent).
- Require plan artifact approval before apply.
- Store immutable plan + apply logs for audit evidence.

## Exercise 3: Delivery workflow design
- Implement trunk-based pipeline with:
  - unit tests
  - integration tests
  - security scans
  - provenance/attestation publish
- Add manual approval for production with segregation-of-duties.

## Exercise 4: Change management evidence
- Build a release checklist capturing:
  - ticket links
  - risk classification
  - rollback steps
  - approvers and timestamps
- Define retention policy for evidence.

## Exercise 5: Incident rollback drill
- Simulate bad deploy and execute rollback within target MTR.
- Capture timeline and corrective actions.
- Produce a one-page postmortem with controls improvement.
