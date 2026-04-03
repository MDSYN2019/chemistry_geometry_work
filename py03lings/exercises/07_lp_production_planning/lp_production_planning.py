"""Exercise 07: multi-period production planning LP with PuLP.

Install once:
    pip install pulp
"""

import pulp


def plan_production() -> dict[str, float]:
    """Solve and return decision values keyed by variable name."""
    months = ["m1", "m2", "m3"]
    demand = {"m1": 30, "m2": 40, "m3": 35}
    prod_cost = {"m1": 5, "m2": 4, "m3": 6}
    hold_cost = 1
    max_prod = 50

    # TODO: create minimization problem "production_planning"
    model = None

    # TODO: create non-negative production vars prod[m]
    prod = None

    # TODO: create non-negative inventory vars inv[m]
    inv = None

    # TODO: objective = sum(prod_cost[m]*prod[m] + hold_cost*inv[m])

    # Inventory balance:
    # inv[m1] = prod[m1] - demand[m1]
    # inv[m2] = inv[m1] + prod[m2] - demand[m2]
    # inv[m3] = inv[m2] + prod[m3] - demand[m3]
    # TODO: encode these constraints

    # TODO: add production capacity constraints prod[m] <= max_prod

    # TODO: solve silently with CBC

    # TODO: return a dict containing prod_*, inv_*, and total_cost
    return {}


if __name__ == "__main__":
    result = plan_production()
    for k, v in result.items():
        print(f"{k}: {v:.2f}")
