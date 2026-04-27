# Senior Data Engineering Interview Practice Pack (SQL + Python + Platform Design)

This pack is designed to mirror the kind of work described in your target role: resilient pipelines, warehouse modeling, stakeholder-focused delivery, and time-critical use cases (like fraud).

## How to use this pack
- Work each exercise in three passes:
  1. **Correctness first** (works on sample data)
  2. **Production hardening** (tests, idempotency, observability)
  3. **Scale and trade-offs** (cost, performance, reliability)
- Timebox each challenge (60–120 min), then write a 5-minute debrief: assumptions, trade-offs, and what you would improve.

---

## Dataset setup (used across SQL + Python tasks)

Assume these tables exist in your warehouse:

- `raw_transactions`
  - `event_id` STRING
  - `customer_id` STRING
  - `account_id` STRING
  - `event_time_utc` TIMESTAMP
  - `amount` DECIMAL(18,2)
  - `currency` STRING
  - `merchant_id` STRING
  - `merchant_category` STRING
  - `country_code` STRING
  - `channel` STRING (`card_present`, `ecommerce`, `bank_transfer`)
  - `status` STRING (`authorized`, `settled`, `reversed`)
  - `ingested_at` TIMESTAMP

- `raw_customers`
  - `customer_id` STRING
  - `signup_date` DATE
  - `risk_segment` STRING
  - `kyc_status` STRING
  - `is_active` BOOLEAN
  - `updated_at` TIMESTAMP

- `fx_rates_daily`
  - `rate_date` DATE
  - `base_currency` STRING
  - `quote_currency` STRING
  - `rate` DECIMAL(18,8)

- `merchant_dim`
  - `merchant_id` STRING
  - `merchant_name` STRING
  - `merchant_category` STRING
  - `country_code` STRING
  - `is_high_risk` BOOLEAN
  - `updated_at` TIMESTAMP

### Data quality wrinkles (intentional)
- Duplicate `event_id` rows may arrive due to replay.
- Late-arriving events can be up to 72 hours late.
- `status` can change over time for the same `event_id`.
- FX rate may be missing for weekends/holidays.

---

## Part A — SQL exercises (advanced)

### 1) Build a canonical transaction fact table
**Goal:** produce `fct_transactions_canonical` with one latest row per `event_id`.

**Requirements:**
- Deduplicate by `event_id` using most recent `ingested_at`.
- Keep full status lifecycle history in a separate `fct_transaction_status_history` table.
- Enforce deterministic tie-breaks when `ingested_at` ties.

**Deliverables:**
- SQL DDL + incremental load SQL.
- Brief explanation of idempotency strategy.

**Stretch:**
- Add a test query that proves no duplicate `event_id` in canonical table.

---

### 2) Time-critical fraud feature table (5-minute SLA)
**Goal:** create `fraud_features_5m` with rolling features per transaction event.

**Features to compute:**
- Count of customer transactions in last 5 min, 1 hour, 24 hours.
- Sum amount in base currency in last 24 hours.
- Distinct countries used by customer in last 24 hours.
- Flag if merchant is high risk.
- Time since last successful (`status='settled'`) transaction.

**Requirements:**
- Handle late-arriving events without double counting.
- Feature values must be point-in-time correct as of event timestamp.

**Stretch:**
- Provide two approaches: pure SQL window functions vs pre-aggregated helper tables.

---

### 3) Slowly changing dimension (SCD Type 2) for customers
**Goal:** implement `dim_customer_scd2`.

**Requirements:**
- Track changes in `risk_segment`, `kyc_status`, `is_active`.
- Include `valid_from`, `valid_to`, `is_current`.
- Support backfilled updates.

**Stretch:**
- Write a query to join transactions to the correct customer version at event time.

---

### 4) Warehouse modeling challenge
**Goal:** model for analytics + operational fraud use cases.

**Prompt:**
- Propose star schema vs data vault vs hybrid for this domain.
- Identify grain for each fact table.
- Define conformed dimensions.
- Explain partitioning/clustering keys.

**Stretch:**
- Explain how your model supports both dashboard latency and ad-hoc deep dives.

---

### 5) Reconciliation and data quality checks
**Goal:** build SQL checks suitable for automated pipeline gates.

**Checks:**
- Freshness (`max(ingested_at)` threshold).
- Volume anomaly (z-score or robust median-based rule).
- Referential integrity (merchant/customer keys).
- Null constraints on critical fields.
- Duplicate and status-transition validity checks.

**Stretch:**
- Add severity levels (`warn`, `error`) and routing recommendations.

---

### 6) Performance tuning scenario
**Problem:** daily job regressed from 15 minutes to 2 hours.

**Task:**
- Diagnose likely causes from SQL patterns.
- Rewrite one expensive query using two alternative patterns.
- Propose indexing/partitioning/clustering improvements.
- Define an experiment plan to prove improvement.

---

## Part B — Python pipeline exercises

### 7) Incremental ingestion pipeline with exactly-once semantics (practical)
**Goal:** implement a Python batch/stream micro-pipeline.

**Requirements:**
- Read events from `raw_transactions` extract files (JSONL/CSV).
- Deduplicate with stable key strategy.
- Upsert into canonical store.
- Maintain watermark/checkpoint state.
- Re-run safety: no duplicates on retries.

**Suggested structure:**
- `extract.py`, `transform.py`, `load.py`, `state_store.py`, `main.py`.

**Stretch:**
- Simulate crash mid-run and demonstrate safe recovery.

---

### 8) Data contract + schema evolution guardrails
**Goal:** detect breaking producer changes before they damage downstream models.

**Requirements:**
- Define schema contract (Pydantic or similar).
- Validate incoming payloads.
- Route bad records to quarantine with reason codes.
- Produce daily contract report.

**Stretch:**
- Implement backward-compatible column add handling without full redeploy.

---

### 9) Pipeline observability instrumentation
**Goal:** make failures diagnosable in under 10 minutes.

**Requirements:**
- Emit structured logs with correlation IDs.
- Emit metrics: throughput, lag, error rate, retry count, DQ failures.
- Add tracing around major stages.
- Produce an “on-call runbook” snippet: symptom → likely cause → first action.

**Stretch:**
- Add SLOs and alert thresholds for each critical metric.

---

### 10) Robust test strategy for data pipelines
**Goal:** create a realistic automated test pyramid.

**Requirements:**
- Unit tests for transformation logic (edge cases + nulls + timezone).
- Contract tests for schema compatibility.
- Integration test with ephemeral local DB.
- Golden dataset regression test.

**Stretch:**
- Add property-based tests for aggregation invariants.

---

### 11) Orchestration + dependency challenge
**Goal:** design DAG for raw ingest → canonical → marts → fraud features.

**Requirements:**
- Task dependency map with retries, timeouts, and backfills.
- Idempotency notes per task.
- Failure strategy (partial failures, reruns, alerting).

**Stretch:**
- Compare Airflow vs Dagster vs managed cloud orchestrator for this use case.

---

### 12) Cost-performance optimization exercise
**Goal:** lower warehouse + compute spend by 30% with no SLA breach.

**Task:**
- Identify top 3 cost drivers.
- Propose quick wins and structural changes.
- Estimate impact/risk/effort for each.
- Define post-change monitoring to avoid regressions.

---

## Part C — End-to-end architecture and leadership scenarios

### 13) Design a near-real-time fraud feature platform
**Prompt:**
Design ingestion, compute, storage, feature serving, and monitoring components.

**Must include:**
- Latency budget (end-to-end)
- Backfill strategy
- Handling retractions/reversals
- Exactly-once/at-least-once trade-offs
- Data governance and PII handling

**Interview angle:**
Communicate trade-offs clearly to non-engineering stakeholders.

---

### 14) Incident response simulation
**Scenario:**
Fraud features are 40 minutes delayed during peak traffic. Analysts report wrong numbers in dashboard.

**Task:**
- Draft incident triage checklist.
- Define containment, diagnosis, and communication plan.
- Provide 24-hour and 2-week remediation plan.

**Stretch:**
- Add “what metrics would have caught this earlier?” section.

---

### 15) Cross-functional requirement gathering role-play
**Task:**
Write a one-page discovery doc with:
- problem statement,
- measurable success criteria,
- known unknowns,
- MVP scope,
- data quality risks,
- rollout plan.

This mimics pairing with Product, Analysts, and business stakeholders.

---

## Part D — Intensive mock interview drills

### Drill 1 (75 minutes): SQL whiteboard + optimization
1. Build canonical dedup model.
2. Add 24-hour rolling fraud aggregates.
3. Explain performance tuning choices.

### Drill 2 (90 minutes): Python pipeline live coding
1. Implement incremental loader.
2. Add schema validation and quarantine path.
3. Add tests for one edge case and one failure path.

### Drill 3 (60 minutes): System design
1. Draw architecture for fraud + analytics dual-use platform.
2. Discuss SLAs/SLOs, lineage, governance, and observability.
3. Defend trade-offs under cost constraints.

---

## Self-evaluation rubric (use after every exercise)

Score each 1–5:
- Correctness
- Reliability/idempotency
- Performance/scalability
- Observability/operability
- Data modeling quality
- Communication/trade-off clarity

If any score is ≤3, write one concrete improvement and re-run.

---

## 4-week practice plan (optional)

### Week 1
- SQL 1, 3, 5
- Python 7 (basic)

### Week 2
- SQL 2, 6
- Python 8, 10

### Week 3
- Python 9, 11
- Architecture 13

### Week 4
- Incident 14
- Role-play 15
- Drills 1–3 under timed conditions

---

## Bonus: realistic interview prompts you can rehearse aloud
- “How do you guarantee trust in data consumed by executives and fraud systems at the same time?”
- “Tell me about a pipeline failure you owned end-to-end. What changed after?”
- “Where would you accept eventual consistency, and where would you not?”
- “How do you balance shipping quickly with long-term maintainability?”
- “When analysts ask for conflicting metrics, how do you resolve source-of-truth disputes?”

Use the STAR method, but include technical depth (scale, latency, test strategy, and operational outcomes).
