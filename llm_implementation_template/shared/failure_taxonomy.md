# Failure Taxonomy Template

Use this list when labeling errors in evaluations and production incidents.

- **Retrieval Miss**: relevant evidence exists but was not retrieved.
- **Context Misread**: evidence was retrieved but interpreted incorrectly.
- **Hallucination**: unsupported claim not grounded in context.
- **Instruction Non-Compliance**: ignored system/developer/user constraints.
- **Tool Misuse**: wrong tool selected or malformed tool arguments.
- **Schema Violation**: structured output invalid or missing required fields.
- **Policy/Safety Violation**: response violates policy requirements.
- **Latency Timeout**: response exceeded SLA or hard timeout.
- **Cost Overrun**: request exceeds token/cost budget.
