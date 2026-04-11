# SQL Practice Set: Performance Optimization + Complex Queries

These exercises are designed for strong SQL learners who want to sharpen:
- Query planning and performance tuning
- Window functions and advanced aggregations
- CTEs (including recursive CTEs)
- Subquery rewrites and anti-join patterns
- Index strategy and execution-plan reasoning

---

## Shared Schema (Use for all exercises)

```sql
CREATE TABLE customers (
  customer_id BIGINT PRIMARY KEY,
  signup_date DATE NOT NULL,
  country_code CHAR(2) NOT NULL,
  segment TEXT NOT NULL
);

CREATE TABLE orders (
  order_id BIGINT PRIMARY KEY,
  customer_id BIGINT NOT NULL REFERENCES customers(customer_id),
  order_ts TIMESTAMP NOT NULL,
  status TEXT NOT NULL,
  total_amount NUMERIC(12,2) NOT NULL,
  currency CHAR(3) NOT NULL
);

CREATE TABLE order_items (
  order_id BIGINT NOT NULL REFERENCES orders(order_id),
  product_id BIGINT NOT NULL,
  quantity INT NOT NULL,
  unit_price NUMERIC(10,2) NOT NULL,
  PRIMARY KEY(order_id, product_id)
);

CREATE TABLE products (
  product_id BIGINT PRIMARY KEY,
  category_id BIGINT NOT NULL,
  sku TEXT NOT NULL,
  active BOOLEAN NOT NULL,
  launched_at DATE NOT NULL
);

CREATE TABLE categories (
  category_id BIGINT PRIMARY KEY,
  category_name TEXT NOT NULL
);

CREATE TABLE page_views (
  view_id BIGINT PRIMARY KEY,
  customer_id BIGINT,
  session_id TEXT NOT NULL,
  page_ts TIMESTAMP NOT NULL,
  page_type TEXT NOT NULL,
  product_id BIGINT
);
```

---

## Part A — Complex Query Construction

### 1) Top-N products per category by rolling 90-day revenue
Return the **top 3 products** in each category by revenue over the last 90 days.

**Requirements**
- Include `category_name`, `product_id`, `revenue_90d`, and rank.
- Use a window function.
- Exclude canceled/refunded orders.

---

### 2) Cohort retention matrix (monthly)
Build a retention view where:
- Cohort month = customer signup month
- Activity month = month with at least one completed order
- Output: `cohort_month`, `month_offset`, `retained_customers`, `cohort_size`, `retention_rate`

**Stretch**: Pivot month offsets 0..6 into columns.

---

### 3) Funnel drop-off from page_views
Create a query for conversion funnel by date:
1. Viewed product page
2. Added to cart page
3. Checkout page
4. Completed order

**Requirements**
- Daily funnel counts and step conversion rates
- Deduplicate by customer-day
- Handle null `customer_id` rows explicitly

---

### 4) First vs repeat purchase behavior
For each country and segment, report:
- Avg order value for first order
- Avg order value for repeat orders
- % customers with repeat orders

**Requirements**
- Correct first order identification by `order_ts`
- Tie-breaking if same timestamp

---

### 5) Gap-and-island for customer activity streaks
Find each customer’s longest streak of active weeks (at least one completed order in a week).

**Output**
- `customer_id`, `longest_streak_weeks`, `streak_start`, `streak_end`

---

## Part B — Performance Optimization Drills

For each prompt:
1) Write an initial query.
2) Inspect EXPLAIN / EXPLAIN ANALYZE.
3) Optimize query and/or indexing.
4) Explain why performance improved.

### 6) Non-sargable predicate rewrite
Initial pattern:
```sql
WHERE DATE(order_ts) = DATE '2026-01-15'
```
Rewrite to be index-friendly and compare plans.

---

### 7) Correlated subquery to join/window rewrite
Given a query that computes per-customer latest order using a correlated subquery, rewrite using:
- window function (`ROW_NUMBER`) approach
- aggregated join approach

Compare timing on large data.

---

### 8) `NOT IN` pitfalls vs `NOT EXISTS`
Find customers who never placed a completed order.
- Implement with `NOT IN`.
- Implement with `NOT EXISTS`.
- Show behavior when nulls are present.

---

### 9) Covering/composite index design
Optimize:
- Recent orders by customer (`customer_id`, date range)
- Top products by category over recent period (join-heavy)

Propose indexes and justify column order.

---

### 10) Pre-aggregation strategy
A dashboard query scans huge `order_items` daily.
Design a summary table/materialized view for daily product revenue.

**Tasks**
- Define schema
- Write incremental refresh SQL
- Show query rewrite using summary table

---

## Part C — Advanced / Interview-Style Challenges

### 11) Recursive CTE hierarchy rollup
Assume categories become hierarchical (`parent_category_id`).
Write recursive CTE to compute subtree revenue for each top-level category.

---

### 12) Percentiles + outlier flagging
For each category/month, compute:
- median order item extended price
- p90
- outlier flag for values > p90 * 1.5

---

### 13) Session-to-order attribution
Attribute each order to the most recent prior product page view in the same session within 24h.

**Requirements**
- one-to-one attribution
- no forward-looking leakage

---

### 14) Slowly changing dimension-style snapshot
Given changing customer segment over time, build query that reports order revenue by segment **as of order date**.

---

### 15) Idempotent upsert pattern
Design an upsert for daily aggregates that can be safely rerun.
Include conflict target and update logic.

---

## Suggested Evaluation Rubric

Score each exercise 0–3 on:
- Correctness
- Performance quality
- Readability/maintainability
- Edge-case handling (nulls, ties, duplicates)

Target: 30+ points across 15 exercises.

---

## Optional Data-Volume Scenarios (for realistic tuning)

- `orders`: 100M rows
- `order_items`: 500M rows
- `page_views`: 1B rows
- Skew: top 1% customers produce 25% of orders

Use these to test cardinality-estimation and skew-aware indexing choices.
