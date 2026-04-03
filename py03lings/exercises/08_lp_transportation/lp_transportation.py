"""Exercise 08: transportation LP with PuLP.

Install once:
    pip install pulp
"""

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

    # TODO: create minimization model "transportation"
    model = None

    # TODO: create non-negative shipment vars ship[(p, w)]
    ship = None

    # TODO: objective = sum(cost[(p,w)] * ship[(p,w)])

    # TODO: supply constraints for each plant
    # sum_w ship[(p,w)] <= supply[p]

    # TODO: demand constraints for each warehouse
    # sum_p ship[(p,w)] >= demand[w]

    # TODO: solve silently with CBC

    # TODO: return dict with each route and total_cost
    return {}


if __name__ == "__main__":
    answer = optimize_shipping()
    for k, v in answer.items():
        print(f"{k}: {v:.2f}")
