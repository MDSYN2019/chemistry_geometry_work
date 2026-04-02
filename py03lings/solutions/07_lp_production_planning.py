"""Solution 07: multi-period production planning LP with PuLP."""

import pulp


def plan_production() -> dict[str, float]:
    months = ["m1", "m2", "m3"]
    demand = {"m1": 30, "m2": 40, "m3": 35}
    prod_cost = {"m1": 5, "m2": 4, "m3": 6}
    hold_cost = 1
    max_prod = 50

    model = pulp.LpProblem("production_planning", pulp.LpMinimize)

    prod = pulp.LpVariable.dicts("prod", months, lowBound=0)
    inv = pulp.LpVariable.dicts("inv", months, lowBound=0)

    model += pulp.lpSum(prod_cost[m] * prod[m] + hold_cost * inv[m] for m in months)

    model += inv["m1"] == prod["m1"] - demand["m1"]
    model += inv["m2"] == inv["m1"] + prod["m2"] - demand["m2"]
    model += inv["m3"] == inv["m2"] + prod["m3"] - demand["m3"]

    for m in months:
        model += prod[m] <= max_prod

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    result = {f"prod_{m}": float(prod[m].value()) for m in months}
    result.update({f"inv_{m}": float(inv[m].value()) for m in months})
    result["total_cost"] = float(pulp.value(model.objective))
    return result


if __name__ == "__main__":
    result = plan_production()
    for k, v in result.items():
        print(f"{k}: {v:.2f}")
