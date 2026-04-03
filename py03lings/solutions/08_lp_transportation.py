"""Solution 08: transportation LP with PuLP."""

import pulp


def optimize_shipping() -> dict[str, float]:
    plants = ["A", "B"]
    warehouses = ["X", "Y", "Z"]

    supply = {"A": 70, "B": 60}
    demand = {"X": 40, "Y": 50, "Z": 40}

    cost = {
        ("A", "X"): 4,
        ("A", "Y"): 6,
        ("A", "Z"): 8,
        ("B", "X"): 5,
        ("B", "Y"): 4,
        ("B", "Z"): 3,
    }

    model = pulp.LpProblem("transportation", pulp.LpMinimize)

    ship = pulp.LpVariable.dicts(
        "ship",
        [(p, w) for p in plants for w in warehouses],
        lowBound=0,
    )

    model += pulp.lpSum(cost[(p, w)] * ship[(p, w)] for p in plants for w in warehouses)

    for p in plants:
        model += pulp.lpSum(ship[(p, w)] for w in warehouses) <= supply[p]

    for w in warehouses:
        model += pulp.lpSum(ship[(p, w)] for p in plants) >= demand[w]

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    out = {f"ship_{p}_{w}": float(ship[(p, w)].value()) for p in plants for w in warehouses}
    out["total_cost"] = float(pulp.value(model.objective))
    return out


if __name__ == "__main__":
    answer = optimize_shipping()
    for k, v in answer.items():
        print(f"{k}: {v:.2f}")
